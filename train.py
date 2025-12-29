import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# 确保当前目录在 PYTHONPATH 中
sys.path.append(os.getcwd())

from model import PoseidonMoE, MoEConfig
from problems.base import get_dataset  
from trainer import MoETrainer

logger = logging.getLogger(__name__)

# ==========================================
# 1. 参数定义 (Arguments)
# ==========================================

@dataclass
class ModelArguments:
    img_size: int = field(default=128, metadata={"help": "Image resolution"})
    patch_size: int = field(default=4, metadata={"help": "Patch size"})
    embed_dim: int = field(default=128, metadata={"help": "Hidden dimension"})
    text_dim: int = field(default=768, metadata={"help": "Text embedding dimension"})
    
    num_experts: int = field(default=4, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Top K experts"})
    aux_loss_weight: float = field(default=0.01, metadata={"help": "Weight for load balancing loss"})
    loss_type: str = field(default="mse", metadata={"help": "mse or l1"})

    fno_layers: int = field(default=4, metadata={"help": "Layers for FNO"})
    mlp_layers: int = field(default=4, metadata={"help": "Layers for MLP"})
    swin_layers: int = field(default=4, metadata={"help": "Layers for Swin"})
    oformer_layers: int = field(default=4, metadata={"help": "Layers for OFormer"})
    kno_layers: int = field(default=4, metadata={"help": "Layers for KNO"})
    wno_layers: int = field(default=4, metadata={"help": "Layers for WNO"})
    fno_modes: int = field(default=16, metadata={"help": "Modes for FNO"})
    swin_window_size: int = field(default=8, metadata={"help": "Window size for Swin"})
    swin_num_heads: int = field(default=4, metadata={"help": "Num heads for Swin"})
    mlp_ratio: float = field(default=4.0, metadata={"help": "MLP expansion ratio"})
    
    max_num_channels: int = field(default=256, metadata={"help": "Max number of physical channels"})
    lr_embedding_recovery: float = field(default=1e-3, metadata={"help": "LR for embedding layers"})
    # 【新增】Poseidon 的时间嵌入学习率
    lr_time_embedding: float = field(default=None, metadata={"help": "LR for time embedding layers"})


@dataclass
class DataArguments:
    dataset_name: str = field(default="fluids.incompressible.Gaussians", metadata={"help": "Dataset name"})
    data_path: str = field(default="./data", metadata={"help": "Path to data files"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit train samples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit eval samples"})
    ar_steps: int = field(default=1, metadata={"help": "Autoregressive steps"})
    
    # 【新增】Poseidon 对齐的数据集超参数
    max_num_train_time_steps: Optional[int] = field(default=None, metadata={"help": "Max time steps for training"})
    train_time_step_size: Optional[int] = field(default=None, metadata={"help": "Time step size for training"})
    train_small_time_transition: bool = field(default=False, metadata={"help": "Train only for next step prediction (delta t=1)"})
    move_data: Optional[str] = field(default=None, metadata={"help": "Move data to scratch"})

# ==========================================
# 2. 自定义 Collator - 增加 time 处理
# ==========================================

def variable_channel_collator(batch):
    if len(batch) == 0:
        return {}
        
    pixel_values = [b['pixel_values'] for b in batch]
    max_c = max(p.shape[0] for p in pixel_values)
    
    B = len(batch)
    H, W = pixel_values[0].shape[1], pixel_values[0].shape[2]
    
    padded_pixels = torch.zeros(B, max_c, H, W)
    padded_masks = torch.zeros(B, max_c)
    padded_channel_ids = torch.zeros((B, max_c), dtype=torch.long)
    
    text_embeddings = []
    labels = []
    times = []  # 【新增】收集 time
    
    for i, item in enumerate(batch):
        c = item['pixel_values'].shape[0]
        padded_pixels[i, :c, :, :] = item['pixel_values']
        padded_masks[i, :c] = 1.0
        
        if 'channel_ids' in item:
            padded_channel_ids[i, :c] = item['channel_ids']
        else:
            padded_channel_ids[i, :c] = torch.arange(c)

        text_embeddings.append(item['text_embedding'])
        # 【新增】收集 time，如果是 None 则默认为 0.0 (稳健性)
        times.append(item.get('time', 0.0))
        
        if 'labels' in item:
             padded_lbl = torch.zeros(max_c, H, W)
             padded_lbl[:c] = item['labels']
             labels.append(padded_lbl)
             
    batch_dict = {
        "pixel_values": padded_pixels,
        "pixel_mask": padded_masks,
        "channel_ids": padded_channel_ids,
        "text_embedding": torch.stack(text_embeddings),
        "time": torch.tensor(times, dtype=torch.float32), # 【新增】打包 time
        "labels": torch.stack(labels) if len(labels) > 0 else None
    }
    return batch_dict

# ==========================================
# 3. 主函数 (Main)
# ==========================================

def main():
    # 1. 参数解析
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        import yaml
        yaml_file = os.path.abspath(sys.argv[1])
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        model_args, data_args, training_args = parser.parse_dict(config_dict)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 初始化 Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()
    
    # 3. 设置随机种子
    set_seed(training_args.seed)

    # 4. 加载数据集 - 【关键修改】处理 Poseidon 风格的超参数
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    # 构造传递给 get_dataset 的参数字典
    dataset_kwargs = {}
    if data_args.max_num_train_time_steps is not None:
        dataset_kwargs["max_num_time_steps"] = data_args.max_num_train_time_steps
    if data_args.train_time_step_size is not None:
        dataset_kwargs["time_step_size"] = data_args.train_time_step_size
    if data_args.train_small_time_transition:
        dataset_kwargs["allowed_time_transitions"] = [1] # 强制只学习单步
    if data_args.move_data is not None:
        dataset_kwargs["move_to_local_scratch"] = data_args.move_data

    # 获取 Train Set
    train_dataset = get_dataset(
        data_args.dataset_name,
        which="train",
        data_path=data_args.data_path,
        num_trajectories=data_args.max_train_samples if data_args.max_train_samples else -1,
        **dataset_kwargs # 传入处理后的参数
    )
    
    # 获取 Eval Set
    eval_dataset = get_dataset(
        data_args.dataset_name,
        which="val",
        data_path=data_args.data_path,
        num_trajectories=data_args.max_eval_samples if data_args.max_eval_samples else -1,
        **dataset_kwargs
    )

    # 5. 初始化模型
    config = MoEConfig(
        img_size=model_args.img_size,
        patch_size=model_args.patch_size,
        embed_dim=model_args.embed_dim,
        text_dim=model_args.text_dim,
        max_num_channels=model_args.max_num_channels, 
        num_experts=model_args.num_experts,
        top_k=model_args.top_k,
        fno_modes=model_args.fno_modes,
        fno_layers=model_args.fno_layers,
        mlp_layers=model_args.mlp_layers,
        swin_layers=model_args.swin_layers,
        oformer_layers=model_args.oformer_layers,
        kno_layers=model_args.kno_layers,
        wno_layers=model_args.wno_layers,
        swin_window_size=model_args.swin_window_size,
        swin_num_heads=model_args.swin_num_heads,
        mlp_ratio=model_args.mlp_ratio
    )
    config.loss_type = model_args.loss_type 

    logger.info("Initializing PoseidonMoE Model...")
    model = PoseidonMoE(config)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")

    # 注入 Trainer 需要的额外参数
    training_args.ar_steps = data_args.ar_steps
    training_args.learning_rate_embedding_recovery = model_args.lr_embedding_recovery
    training_args.learning_rate_time_embedding = model_args.lr_time_embedding # 【新增】
    training_args.label_names = ["labels"]

    # 6. 初始化 Trainer
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_args=model_args, 
        data_collator=variable_channel_collator, 
        aux_loss_weight=model_args.aux_loss_weight,
    )

    # 7. 断点检测逻辑
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warning(f"Output directory {training_args.output_dir} is not empty.")

    # 8. 开始训练
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 9. 评估
    if training_args.do_eval:
        logger.info("*** Starting Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()

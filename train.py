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
# 【新增】引入检查点工具
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
    
    fno_modes: int = field(default=16, metadata={"help": "Modes for FNO"})
    swin_window_size: int = field(default=8, metadata={"help": "Window size for Swin"})
    swin_num_heads: int = field(default=4, metadata={"help": "Num heads for Swin"})
    mlp_ratio: float = field(default=4.0, metadata={"help": "MLP expansion ratio"})
    
    max_num_channels: int = field(default=256, metadata={"help": "Max number of physical channels"})
    lr_embedding_recovery: float = field(default=1e-3, metadata={"help": "LR for embedding layers"})


@dataclass
class DataArguments:
    dataset_name: str = field(default="fluids.incompressible.Gaussians", metadata={"help": "Dataset name"})
    data_path: str = field(default="./data", metadata={"help": "Path to data files"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit train samples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit eval samples"})
    ar_steps: int = field(default=1, metadata={"help": "Autoregressive steps"})

# ==========================================
# 2. 自定义 Collator
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
    
    for i, item in enumerate(batch):
        c = item['pixel_values'].shape[0]
        padded_pixels[i, :c, :, :] = item['pixel_values']
        padded_masks[i, :c] = 1.0
        
        if 'channel_ids' in item:
            padded_channel_ids[i, :c] = item['channel_ids']
        else:
            padded_channel_ids[i, :c] = torch.arange(c)

        text_embeddings.append(item['text_embedding'])
        
        if 'labels' in item:
             padded_lbl = torch.zeros(max_c, H, W)
             padded_lbl[:c] = item['labels']
             labels.append(padded_lbl)
             
    batch_dict = {
        "pixel_values": padded_pixels,
        "pixel_mask": padded_masks,
        "channel_ids": padded_channel_ids,
        "text_embedding": torch.stack(text_embeddings),
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

    # 4. 加载数据集
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    train_dataset = get_dataset(
        data_args.dataset_name,
        which="train",
        data_path=data_args.data_path,
        num_trajectories=data_args.max_train_samples if data_args.max_train_samples else -1,
    )
    
    eval_dataset = get_dataset(
        data_args.dataset_name,
        which="val",
        data_path=data_args.data_path,
        num_trajectories=data_args.max_eval_samples if data_args.max_eval_samples else -1
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

    # ====================================================
    # 7. 断点检测逻辑 (仅用于日志提示，不强制续训)
    # ====================================================
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warning(
                f"Output directory ({training_args.output_dir}) exists and is not empty. "
                "No valid checkpoint found. Training might fail if overwrite is not allowed."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected at {last_checkpoint}. "
                f"To resume training, use '--resume_from_checkpoint' flag."
            )

    # 8. 开始训练
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        
        # 【关键修改】
        # 不再自动传入 last_checkpoint。
        # 如果命令行加了 --resume_from_checkpoint，training_args.resume_from_checkpoint 会自动生效。
        # 如果没加，则从头开始（或因目录非空且未 overwrite 而报错，符合预期）。
        
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
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

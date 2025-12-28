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

from model import PoseidonMoE, MoEConfig
from problems.base import get_dataset  
from trainer import MoETrainer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型架构相关参数
    """
    img_size: int = field(default=128, metadata={"help": "Image resolution"})
    patch_size: int = field(default=4, metadata={"help": "Patch size"})
    embed_dim: int = field(default=128, metadata={"help": "Hidden dimension"})
    text_dim: int = field(default=768, metadata={"help": "Text embedding dimension"})
    
    num_experts: int = field(default=4, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Top K experts"})
    fno_modes: int = field(default=16, metadata={"help": "Modes for FNO"})
    
    aux_loss_weight: float = field(default=0.01, metadata={"help": "Weight for load balancing loss"})
    loss_type: str = field(default="mse", metadata={"help": "mse or l1"})
    
    # MLP / Swin 特定参数
    swin_window_size: int = field(default=8, metadata={"help": "Window size for Swin"})
    swin_num_heads: int = field(default=4, metadata={"help": "Num heads for Swin"})
    mlp_ratio: float = field(default=4.0, metadata={"help": "MLP expansion ratio"})
    
    # 物理通道与学习率
    max_num_channels: int = field(default=256, metadata={"help": "Max number of physical channels"})
    lr_embedding_recovery: float = field(default=1e-3, metadata={"help": "LR for embedding layers"})


@dataclass
class DataArguments:
    """
    数据加载相关参数
    """
    dataset_name: str = field(default="fluids.incompressible.Gaussians", metadata={"help": "Dataset name"})
    data_path: str = field(default="./data", metadata={"help": "Path to data files"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit train samples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit eval samples"})
    ar_steps: int = field(default=1, metadata={"help": "Autoregressive steps"})

def main():
    # 1. 定义解析器
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # 2. 解析参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        import yaml
        yaml_file = os.path.abspath(sys.argv[1])
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        # 过滤 None
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        model_args, data_args, training_args = parser.parse_dict(config_dict)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 3. Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 4. Set Seed
    set_seed(training_args.seed)

    # 5. Load Datasets
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    # 【修复 1】：从这里移除了 ar_steps=...，因为 dataset 类不接受它
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

    # 6. Initialize Config & Model
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
    # 动态注入参数
    config.loss_type = model_args.loss_type 

    logger.info("Initializing PoseidonMoE Model...")
    model = PoseidonMoE(config)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")

    # 【修复 2】：将 ar_steps 注入到 training_args 中，因为 Trainer 默认从 args 里读
    training_args.ar_steps = data_args.ar_steps
    # 注入 learning_rate_embedding_recovery 供 Trainer 优化器分组使用
    training_args.learning_rate_embedding_recovery = model_args.lr_embedding_recovery

    # 7. Initialize Custom Trainer
    from train import variable_channel_collator # 确保 collator 被导入或定义
    
    # 如果 variable_channel_collator 定义在 train.py 里，直接用即可
    # 否则请将其加上
    
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # model_args=model_args, # 如果你的 Trainer 构造函数支持 model_args 则加上
        data_collator=variable_channel_collator, 
        aux_loss_weight=model_args.aux_loss_weight,
    )

    # 8. Start Training
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 9. Evaluation
    if training_args.do_eval:
        logger.info("*** Starting Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

# --------------------------------------------------------
# 补充定义 Collator (如果之前漏了)
# --------------------------------------------------------
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

if __name__ == "__main__":
    main()

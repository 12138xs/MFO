# import argparse
# import torch
# import wandb
# import numpy as np
# import random
# import json
# import psutil
# import os

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# import yaml
# import matplotlib.pyplot as plt
# import transformers
# from accelerate.utils import broadcast_object_list
# from transformers import EarlyStoppingCallback
# from mpl_toolkits.axes_grid1 import ImageGrid

# from problems.base import get_dataset, BaseTimeDataset
# from utils import get_num_parameters, read_cli, get_num_parameters_no_embed
# from metrics import relative_lp_error

# SEED = 0
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)



# def create_predictions_plot(predictions, labels, wandb_prefix):
#     assert predictions.shape[0] >= 4

#     indices = random.sample(range(predictions.shape[0]), 4)

#     predictions = predictions[indices]
#     labels = labels[indices]

#     fig = plt.figure()
#     grid = ImageGrid(
#         fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
#     )

#     vmax, vmin = max(predictions.max(), labels.max()), min(
#         predictions.min(), labels.min()
#     )

#     for _i, ax in enumerate(grid):
#         i = _i // 4
#         j = _i % 4

#         if i % 2 == 0:
#             ax.imshow(
#                 predictions[j, i // 2, :, :],
#                 cmap="gist_ncar",
#                 origin="lower",
#                 vmin=vmin,
#                 vmax=vmax,
#             )
#         else:
#             ax.imshow(
#                 labels[j, i // 2, :, :],
#                 cmap="gist_ncar",
#                 origin="lower",
#                 vmin=vmin,
#                 vmax=vmax,
#             )

#         ax.set_xticks([])
#         ax.set_yticks([])

#     wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
#     plt.close()


# def setup(params, model_map=True):
#     config = None
#     RANK = int(os.environ.get("LOCAL_RANK", -1))
#     CPU_CORES = len(psutil.Process().cpu_affinity())
#     CPU_CORES = min(CPU_CORES, 16)
#     print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
#     if params.disable_tqdm:
#         transformers.utils.logging.disable_progress_bar()
#     if params.json_config:
#         config = json.loads(params.config)
#     else:
#         config = params.config

#     if RANK == 0 or RANK == -1:
#         run = wandb.init(
#             project=params.wandb_project_name, name=params.wandb_run_name, config=config
#         )
#         config = wandb.config
#     else:

#         def clean_yaml(config):
#             d = {}
#             for key, inner_dict in config.items():
#                 d[key] = inner_dict["value"]
#             return d

#         if not params.json_config:
#             with open(params.config, "r") as s:
#                 config = yaml.safe_load(s)
#             config = clean_yaml(config)
#         run = None

#     ckpt_dir = "./"
#     if RANK == 0 or RANK == -1:
#         if run.sweep_id is not None:
#             ckpt_dir = (
#                 params.checkpoint_path
#                 + "/"
#                 + run.project
#                 + "/"
#                 + run.sweep_id
#                 + "/"
#                 + run.name
#             )
#         else:
#             ckpt_dir = params.checkpoint_path + "/" + run.project + "/" + run.name
#     if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     ls = broadcast_object_list([ckpt_dir], from_process=0)
#     ckpt_dir = ls[0]

#     return run, config, ckpt_dir, RANK, CPU_CORES


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train scOT.")
#     params = read_cli(parser).parse_args()
#     run, config, ckpt_dir, RANK, CPU_CORES = setup(params)

#     train_eval_set_kwargs = (
#         {"just_velocities": True}
#         if ("incompressible" in config["dataset"]) and params.just_velocities
#         else {}
#     )
#     if params.move_data is not None:
#         train_eval_set_kwargs["move_to_local_scratch"] = params.move_data
#     if params.max_num_train_time_steps is not None:
#         train_eval_set_kwargs["max_num_time_steps"] = params.max_num_train_time_steps
#     if params.train_time_step_size is not None:
#         train_eval_set_kwargs["time_step_size"] = params.train_time_step_size
#     if params.train_small_time_transition:
#         train_eval_set_kwargs["allowed_time_transitions"] = [1]
#     train_dataset = get_dataset(
#         dataset=config["dataset"],
#         which="train",
#         num_trajectories=config["num_trajectories"],
#         data_path=params.data_path,
#         **train_eval_set_kwargs,
#     )
#     eval_dataset = get_dataset(
#         dataset=config["dataset"],
#         which="val",
#         num_trajectories=config["num_trajectories"],
#         data_path=params.data_path,
#         **train_eval_set_kwargs,
#     )

#     config["effective_train_set_size"] = len(train_dataset)
#     time_involved = isinstance(train_dataset, BaseTimeDataset) or (
#         isinstance(train_dataset, torch.utils.data.ConcatDataset)
#         and isinstance(train_dataset.datasets[0], BaseTimeDataset)
#     )

#     if not isinstance(train_dataset, torch.utils.data.ConcatDataset):
#         resolution = train_dataset.resolution
#         input_dim = train_dataset.input_dim
#         output_dim = train_dataset.output_dim
#         channel_slice_list = train_dataset.channel_slice_list
#         printable_channel_description = train_dataset.printable_channel_description
#     else:
#         resolution = train_dataset.datasets[0].resolution
#         input_dim = train_dataset.datasets[0].input_dim
#         output_dim = train_dataset.datasets[0].output_dim
#         channel_slice_list = train_dataset.datasets[0].channel_slice_list
#         printable_channel_description = train_dataset.datasets[0].printable_channel_description

#     print("Train dataset info:")
#     print(f"Resolution: {resolution}")
#     print(f"Input dimension: {input_dim}")
#     print(f"Output dimension: {output_dim}")
#     print(f"Channel description: {printable_channel_description}")

#     print(train_dataset)

#     for case in iter(train_dataset):
#         print("One data case info:")
#         print(case.keys())
#         # {
#         #     "pixel_values": inputs,
#         #     "labels": label,
#         #     "time": time,
#         #     "pixel_mask": self.pixel_mask,
#         #     "text_embedding": text_embedding,
#         # }
#         print(f"Inputs shape: {case['pixel_values'].shape}")
#         print(f"Labels shape: {case['labels'].shape}")
#         if time_involved:
#             print(f"Time steps: {case['time']}")
#         if "pixel_mask" in case:
#             print(f"Pixel mask shape: {case['pixel_mask'].shape}")
#         if "text_embedding" in case:
#             print(f"Text embedding shape: {case['text_embedding'].shape}")
#         break



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
    num_experts: int = field(default=4, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Top K experts"})
    fno_modes: int = field(default=16, metadata={"help": "Modes for FNO"})
    aux_loss_weight: float = field(default=0.01, metadata={"help": "Weight for load balancing loss"})
    loss_type: str = field(default="mse", metadata={"help": "mse or l1"})
    # 注意：max_train_samples 不在这里

@dataclass
class DataArguments:
    """
    数据加载相关参数
    """
    dataset_name: str = field(default="fluids.incompressible.Gaussians", metadata={"help": "Dataset name"})
    data_path: str = field(default="./data", metadata={"help": "Path to data files"})
    # 这些参数定义在这里：
    max_train_samples: int = field(default=None, metadata={"help": "Limit train samples"})
    max_eval_samples: int = field(default=None, metadata={"help": "Limit eval samples"})
    ar_steps: int = field(default=1, metadata={"help": "Autoregressive steps"})

def main():
    # 1. 定义解析器
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # 2. 解析参数 (支持 yaml, json, 命令行)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        import yaml
        yaml_file = os.path.abspath(sys.argv[1])
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
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
    
    # 【修复点 1】：使用 data_args 而不是 model_args
    train_dataset = get_dataset(
        data_args.dataset_name,
        which="train",
        data_path=data_args.data_path,
        num_trajectories=data_args.max_train_samples if data_args.max_train_samples else -1,
        ar_steps=data_args.ar_steps # 确保 base.py 里的 get_dataset 接受 ar_steps 参数，或者在此处处理
    )
    
    # 【修复点 2】：使用 data_args 而不是 model_args
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
        num_experts=model_args.num_experts,
        top_k=model_args.top_k,
        fno_modes=model_args.fno_modes,
    )
    # 动态注入参数
    config.loss_type = model_args.loss_type 

    logger.info("Initializing PoseidonMoE Model...")
    model = PoseidonMoE(config)
    
    # 打印参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")

    # 7. Initialize Custom Trainer
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

if __name__ == "__main__":
    main()

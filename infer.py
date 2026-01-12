import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

# 确保能导入 MFO 模块
sys.path.append(os.getcwd())

try:
    from model import PoseidonMoE, MoEConfig
    from problems.base import get_dataset
    from metrics import relative_lp_error
except ImportError:
    print("Error: Could not import MFO modules. Please run this script from the root of the MFO repository.")
    sys.exit(1)

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================

TASK_CONFIGS = {
    # --- Compressible Fluids ---
    "fluids.compressible.RichtmyerMeshkov": {
        "display_name": "CE-RM",
        "initial_time": 0, "final_time": 20, "ar_steps": 20, "ss_step_size": 1, "is_static": False,
        "dataset_kwargs": {}
    },
    "fluids.compressible.RiemannKelvinHelmholtz": {
        "display_name": "CE-RPUI",
        "initial_time": 0, "final_time": 20, "ar_steps": 20, "ss_step_size": 1, "is_static": False,
        "dataset_kwargs": {}
    },
    # --- Reaction Diffusion ---
    "reaction_diffusion.AllenCahn": {
        "display_name": "ACE",
        "initial_time": 0, "final_time": 19, "ar_steps": 19, "ss_step_size": 1, "is_static": False,
        "dataset_kwargs": {}
    },
    # --- Incompressible Fluids (应用 just_velocities) ---
    "fluids.incompressible.forcing.KolmogorovFlow": {
        "display_name": "FNS-KF",
        "initial_time": 0, "final_time": 20, "ar_steps": 20, "ss_step_size": 1, "is_static": False,
        "dataset_kwargs": {"just_velocities": True} # [Fix] Only use u, v, g (3 channels)
    },
    "fluids.incompressible.BrownianBridge": {
        "display_name": "NS-BB",
        "initial_time": 0, "final_time": 20, "ar_steps": 20, "ss_step_size": 1, "is_static": False,
        "dataset_kwargs": {"just_velocities": True} # [Fix] Only use u, v (2 channels)
    },
    # --- Static / Elliptic ---
    "elliptic.poisson.Gaussians.time": {
        "display_name": "Poisson-Gauss",
        "initial_time": 0, "final_time": 1, "ar_steps": 0, "ss_step_size": 1, "is_static": True,
        "dataset_kwargs": {}
    },
    "elliptic.Helmholtz.time": {
        "display_name": "Helmholtz",
        "initial_time": 0, "final_time": 1, "ar_steps": 0, "ss_step_size": 1, "is_static": True,
        "dataset_kwargs": {}
    }
}

# ==========================================
# 2. Helper Functions & Collator
# ==========================================

def variable_channel_collator(batch):
    """
    Copied from MFO/train.py: Handles batching of variable channel data.
    """
    if len(batch) == 0: return {}
    pixel_values = [b['pixel_values'] for b in batch]
    max_c = max(p.shape[0] for p in pixel_values)
    
    B = len(batch)
    H, W = pixel_values[0].shape[1], pixel_values[0].shape[2]
    
    padded_pixels = torch.zeros(B, max_c, H, W)
    padded_masks = torch.zeros(B, max_c)
    padded_channel_ids = torch.zeros((B, max_c), dtype=torch.long)
    text_embeddings = []
    labels = []
    times = []
    
    for i, item in enumerate(batch):
        c = item['pixel_values'].shape[0]
        padded_pixels[i, :c, :, :] = item['pixel_values']
        padded_masks[i, :c] = 1.0
        
        if 'channel_ids' in item:
            padded_channel_ids[i, :c] = item['channel_ids']
        else:
            padded_channel_ids[i, :c] = torch.arange(c)
            
        text_embeddings.append(item['text_embedding'])
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
        "time": torch.tensor(times, dtype=torch.float32),
        "labels": torch.stack(labels) if len(labels) > 0 else None
    }
    return batch_dict

def max_absolute_error(preds, targets):
    return np.max(np.abs(preds - targets))

def get_trajectories(task_key, data_path, ar_steps, initial_time, final_time, dataset_kwargs={}):
    """
    Manually reconstruct ground truth trajectories for AR evaluation.
    """
    gt_traj = []
    # For each step i from 1 to ar_steps
    for step in range(1, ar_steps + 1):
        target_time = initial_time + step * (final_time - initial_time) // ar_steps
        
        ds = get_dataset(
            dataset=task_key,
            which="test",
            num_trajectories=240, 
            data_path=data_path,
            fix_input_to_time_step=initial_time,
            time_step_size=target_time - initial_time,
            max_num_time_steps=1, # Just need this pair
            **dataset_kwargs # Pass extra args like just_velocities
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=variable_channel_collator, num_workers=4)
        
        step_labels = []
        for batch in loader:
            step_labels.append(batch['labels'])
        gt_traj.append(torch.cat(step_labels, dim=0))
        
    # Stack to (N, T, C, H, W)
    return torch.stack(gt_traj, dim=1)

# ==========================================
# 3. Evaluation Logic
# ==========================================

def evaluate_task(task_key, config, model, data_path, device):
    display_name = config['display_name']
    print(f"\nEvaluating Task: {display_name} ({task_key})")
    
    # Get dataset specific args (e.g. just_velocities)
    ds_extra_kwargs = config.get("dataset_kwargs", {})
    if ds_extra_kwargs:
        print(f"    Applying Dataset Args: {ds_extra_kwargs}")

    # ---------------------------------------------------------
    # PART 1: Single-Step Inference (SS)
    # ---------------------------------------------------------
    print("  -> Running Single-Step Inference...")
    t0 = time.time()
    
    # Load dataset with all single-step transitions
    if not config['is_static']:
        ss_kwargs = {
            "max_num_time_steps": config['final_time'],
            "time_step_size": config['ss_step_size'],
            "allowed_time_transitions": [config['ss_step_size']],
            **ds_extra_kwargs
        }
    else:
        ss_kwargs = {**ds_extra_kwargs}
        
    try:
        ss_dataset = get_dataset(
            dataset=task_key,
            which="test",
            num_trajectories=240,
            data_path=data_path,
            **ss_kwargs
        )
        ss_loader = DataLoader(ss_dataset, batch_size=32, shuffle=False, 
                               num_workers=4, collate_fn=variable_channel_collator)
    except Exception as e:
        print(f"  [ERROR] Failed to load dataset: {e}")
        return

    # Check input dim
    print(f"    SS Dataset Info: {len(ss_dataset)} samples, Input Dim: {ss_dataset.input_dim}")

    ss_preds_list = []
    ss_targets_list = []

    try:
        with torch.no_grad():
            for batch in tqdm(ss_loader, desc="    SS Batch"):
                pixel_values = batch['pixel_values'].to(device)
                pixel_mask = batch['pixel_mask'].to(device)
                text_embedding = batch['text_embedding'].to(device)
                channel_ids = batch['channel_ids'].to(device)
                time_input = batch['time'].to(device)
                labels = batch['labels'].to(device)

                # MFO forward pass
                outputs, _ = model(
                    pixel_values=pixel_values,
                    text_embedding=text_embedding,
                    pixel_mask=pixel_mask,
                    time=time_input,
                    channel_ids=channel_ids
                )
                
                # Filter padded output for correct metrics
                # MFO collator pads to max_c. We should only consider valid channels.
                # However, since we batch similar items (from same dataset), usually all have same C.
                # We can use pixel_mask to zero out or select.
                
                # Masking for safety (metrics should ignore padding)
                # Expand mask: (B, C) -> (B, C, H, W)
                mask_expanded = pixel_mask.unsqueeze(-1).unsqueeze(-1)
                outputs = outputs * mask_expanded
                labels = labels * mask_expanded
                
                ss_preds_list.append(outputs.cpu())
                ss_targets_list.append(labels.cpu())

        ss_preds = torch.cat(ss_preds_list, dim=0).numpy()
        ss_targets = torch.cat(ss_targets_list, dim=0).numpy()
        
        # Compute metrics
        print("  SS preds shape:", ss_preds.shape, "targets shape:", ss_targets.shape)
        ss_mse = np.mean((ss_preds - ss_targets)**2)
        ss_rel_l1 = np.mean(relative_lp_error(ss_preds, ss_targets, p=1))
        ss_rel_l2 = np.mean(relative_lp_error(ss_preds, ss_targets, p=2))
        ss_max_abs = max_absolute_error(ss_preds, ss_targets)
        
    except Exception as e:
        print(f"  [ERROR] SS inference failed: {e}")
        ss_mse, ss_rel_l1, ss_rel_l2, ss_max_abs = None, None, None, None
        import traceback
        traceback.print_exc()

    ss_duration = (time.time() - t0) * 1000

    # ---------------------------------------------------------
    # PART 2: Autoregressive Inference (AR)
    # ---------------------------------------------------------
    ar_mse, ar_rel_l1, ar_rel_l2, ar_max_abs, ar_time_sec = None, None, None, None, None

    if not config['is_static'] and ss_mse is not None:
        print(f"  -> Running Autoregressive Inference ({config['ar_steps']} steps)...")
        t_ar_start = time.time()
        
        # 1. Load Initial Conditions (t=0)
        ar_input_dataset = get_dataset(
            dataset=task_key,
            which="test",
            num_trajectories=240,
            data_path=data_path,
            fix_input_to_time_step=config['initial_time'],
            max_num_time_steps=1, 
            time_step_size=config['ss_step_size'],
            **ds_extra_kwargs
        )
        
        # Determine normalized time step
        inner_ds = ar_input_dataset.dataset if hasattr(ar_input_dataset, 'dataset') else ar_input_dataset
        while hasattr(inner_ds, 'dataset'): inner_ds = inner_ds.dataset
        
        if hasattr(inner_ds, 'constants') and 'time' in inner_ds.constants:
            phys_time_const = inner_ds.constants['time']
            step_time_val = (1.0 * config['ss_step_size']) / phys_time_const
        else:
            print("    [WARN] Could not determine time constant, defaulting to 0.05")
            step_time_val = 0.05

        ar_loader = DataLoader(ar_input_dataset, batch_size=32, shuffle=False, 
                               num_workers=4, collate_fn=variable_channel_collator)
        
        ar_preds_all = []
        
        try:
            for batch in tqdm(ar_loader, desc="    AR Loop"):
                curr_pixels = batch['pixel_values'].to(device)
                pixel_mask = batch['pixel_mask'].to(device)
                text_embedding = batch['text_embedding'].to(device)
                channel_ids = batch['channel_ids'].to(device)
                
                # Time tensor for 1 step
                B = curr_pixels.shape[0]
                time_input = torch.full((B,), step_time_val, device=device, dtype=torch.float32)
                
                batch_traj = []
                
                with torch.no_grad():
                    for _ in range(config['ar_steps']):
                        outputs, _ = model(
                            pixel_values=curr_pixels,
                            text_embedding=text_embedding,
                            pixel_mask=pixel_mask,
                            time=time_input,
                            channel_ids=channel_ids
                        )
                        # Mask output to keep it clean for next step
                        mask_expanded = pixel_mask.unsqueeze(-1).unsqueeze(-1)
                        outputs = outputs * mask_expanded
                        
                        batch_traj.append(outputs.cpu())
                        curr_pixels = outputs 
                
                ar_preds_all.append(torch.stack(batch_traj, dim=1))
                
            ar_preds = torch.cat(ar_preds_all, dim=0) 
            
            # 2. Get Ground Truth Trajectories
            ar_targets = get_trajectories(
                task_key, data_path, 
                config['ar_steps'], 
                config['initial_time'], 
                config['final_time'],
                dataset_kwargs=ds_extra_kwargs # Pass args here too
            )
            
            # 3. Compute Metrics
            min_len = min(len(ar_preds), len(ar_targets))
            ar_preds = ar_preds[:min_len].numpy()
            ar_targets = ar_targets[:min_len].numpy()
            
            ar_time_sec = time.time() - t_ar_start
            
            print("  AR preds shape:", ar_preds.shape, "targets shape:", ar_targets.shape)
            ar_mse = np.mean((ar_preds - ar_targets)**2)
            
            flat_preds = ar_preds.reshape(-1, *ar_preds.shape[2:])
            flat_targets = ar_targets.reshape(-1, *ar_targets.shape[2:])
            
            ar_rel_l1 = np.mean(relative_lp_error(flat_preds, flat_targets, p=1))
            ar_rel_l2 = np.mean(relative_lp_error(flat_preds, flat_targets, p=2))
            ar_max_abs = max_absolute_error(ar_preds, ar_targets)
            
        except Exception as e:
            print(f"  [ERROR] AR inference failed: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------
    # Output Table
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"   RESULTS for {display_name}")
    print("="*70)
    
    headers = ["Metric", "Single-Step", "Autoregressive"]
    row_fmt = "{:<20} | {:<20} | {:<20}"
    def fmt_val(v): return f"{v:.6e}" if v is not None else "N/A"
    
    print(row_fmt.format(*headers))
    print("-" * 70)
    print(row_fmt.format("MSE", fmt_val(ss_mse), fmt_val(ar_mse)))
    print(row_fmt.format("Rel L1", fmt_val(ss_rel_l1), fmt_val(ar_rel_l1)))
    print(row_fmt.format("Rel L2", fmt_val(ss_rel_l2), fmt_val(ar_rel_l2)))
    print(row_fmt.format("Max Abs Error", fmt_val(ss_max_abs), fmt_val(ar_max_abs)))
    print("-" * 70)
    
    avg_ss_time = ss_duration / len(ss_dataset) if (ss_mse is not None and len(ss_dataset) > 0) else 0
    print(f"SS Avg Time/Sample  : {avg_ss_time:.2f} ms")
    if ar_time_sec: print(f"AR Total Time       : {ar_time_sec:.2f} s")
    print("="*70 + "\n")

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to MFO checkpoint")
    parser.add_argument("--data_path", type=str, default="./data", help="Root path for datasets")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    print(f"Starting MFO Evaluation using: {args.checkpoint_dir}")
    print(f"Data Path: {args.data_path}")
    
    try:
        print("Loading Model...", end=" ")
        model = PoseidonMoE.from_pretrained(args.checkpoint_dir)
        model.to(args.device)
        if args.fp16: model.half()
        model.eval()
        print("Done.")
    except Exception as e:
        print(f"\nCritical Error loading model: {e}")
        sys.exit(1)

    tasks_ordered = [
        "fluids.compressible.RichtmyerMeshkov",
        "fluids.compressible.RiemannKelvinHelmholtz",
        "reaction_diffusion.AllenCahn",
        "fluids.incompressible.forcing.KolmogorovFlow",
        "fluids.incompressible.BrownianBridge",
        "elliptic.poisson.Gaussians.time",
        "elliptic.Helmholtz.time"
    ]

    for task_key in tasks_ordered:
        evaluate_task(task_key, TASK_CONFIGS[task_key], model, args.data_path, args.device)

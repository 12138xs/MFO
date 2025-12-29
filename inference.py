import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 确保能导入本地模块
sys.path.append(os.getcwd())

try:
    from model import PoseidonMoE
    from problems.base import get_dataset
except ImportError:
    print("Error: Could not import MFO modules (model.py). Please run this script from the root of the MFO repository.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for MFO (Final)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data_path", type=str, default="./data", help="Root path to data")
    parser.add_argument("--dataset_name", type=str, default="fluids.incompressible.Gaussians", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--ar_steps", type=int, default=1)
    parser.add_argument("--time_step_size", type=int, default=None)
    return parser.parse_args()

def variable_channel_collator(batch):
    """
    Collator from MFO train.py to handle variable channels
    """
    if len(batch) == 0: return {}
    pixel_values = [b['pixel_values'] for b in batch]
    # Find max channels in this batch
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
        # Copy Data
        padded_pixels[i, :c, :, :] = item['pixel_values']
        # Create Mask (1 for valid, 0 for pad)
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

def compute_metrics(preds, targets, mask=None):
    """
    Standard Metric Calculation (Exact copy of Poseidon version)
    """
    if mask is not None:
        # MFO mask shape is [B, C], expand to [B, C, H, W]
        if mask.ndim == 2:
            mask_expanded = mask.view(preds.shape[0], preds.shape[1], 1, 1).expand_as(preds)
        else:
            mask_expanded = mask
        
        preds = preds * mask_expanded
        targets = targets * mask_expanded
        valid_elements = mask_expanded.sum() * preds.shape[-1] * preds.shape[-2]
        valid_elements = torch.clamp(valid_elements, min=1.0)
    else:
        valid_elements = torch.tensor(preds.numel(), device=preds.device)

    # 1. MSE & L1
    mse = F.mse_loss(preds, targets, reduction='sum') / valid_elements
    l1 = F.l1_loss(preds, targets, reduction='sum') / valid_elements
    
    # 2. Relative L2
    B = preds.shape[0]
    diff = (preds - targets).reshape(B, -1)
    target_flat = targets.reshape(B, -1)
    diff_norm = torch.norm(diff, p=2, dim=1)
    target_norm = torch.norm(target_flat, p=2, dim=1)
    rel_l2 = (diff_norm / (target_norm + 1e-8)).mean()
    
    return mse.item(), l1.item(), rel_l2.item()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"=== MFO Inference ===")
    print(f"Model Path: {args.checkpoint_dir}")
    
    # 1. Load Model
    print("Loading model...", end=" ")
    try:
        model = PoseidonMoE.from_pretrained(args.checkpoint_dir)
        model.to(device)
        if args.fp16: model.half()
        model.eval()
        print("Done.")
    except Exception as e:
        print(f"\nError loading model: {e}")
        return
    
    # 2. Load Data
    print("Loading dataset...", end=" ")
    ds_kwargs = {}
    if args.time_step_size is not None:
        ds_kwargs["time_step_size"] = args.time_step_size
        
    dataset = get_dataset(
        dataset=args.dataset_name,
        which=args.split,
        num_trajectories=args.num_samples if args.num_samples > 0 else -1,
        data_path=args.data_path,
        **ds_kwargs
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, collate_fn=variable_channel_collator)
    print(f"Done. Samples: {len(dataset)}")

    # 3. Inference Loop
    metrics_sum = {"mse": 0.0, "l1": 0.0, "rel_l2": 0.0}
    total_time = 0.0
    num_batches = 0
    
    if args.device == 'cuda': torch.cuda.synchronize()
        
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            text_embedding = batch['text_embedding'].to(device)
            channel_ids = batch['channel_ids'].to(device)
            labels = batch['labels'].to(device) if batch['labels'] is not None else None
            
            if labels is None: continue
            
            if args.fp16:
                pixel_values = pixel_values.half()
                text_embedding = text_embedding.half()

            start_t = time.time()
            curr_pixels = pixel_values
            
            # --- AR Loop ---
            for step in range(args.ar_steps):
                # MFO forward
                outputs, _ = model(
                    pixel_values=curr_pixels,
                    pixel_mask=pixel_mask,
                    text_embedding=text_embedding,
                    channel_ids=channel_ids
                )
                
                # Next step input
                if step < args.ar_steps - 1:
                    curr_pixels = outputs
            
            if args.device == 'cuda': torch.cuda.synchronize()
            total_time += (time.time() - start_t)
            
            if args.fp16: outputs = outputs.float()
            
            # Fix label dimensions if needed (Poseidon dataset consistency)
            if labels.ndim > outputs.ndim: 
                labels = labels.squeeze(2)

            # --- Metrics ---
            m_mse, m_l1, m_rel = compute_metrics(outputs, labels, pixel_mask)
            
            metrics_sum["mse"] += m_mse
            metrics_sum["l1"] += m_l1
            metrics_sum["rel_l2"] += m_rel
            num_batches += 1
            
    # 4. Results
    if num_batches > 0:
        print("\n" + "="*30)
        print("   MFO RESULTS")
        print("="*30)
        print(f"MSE       : {metrics_sum['mse']/num_batches:.6e}")
        print(f"L1        : {metrics_sum['l1']/num_batches:.6e}")
        print(f"Rel L2    : {metrics_sum['rel_l2']/num_batches:.6e}")
        print("-" * 30)
        print(f"Avg Time  : {(total_time/num_batches)*1000:.2f} ms/batch")
        print("="*30)

if __name__ == "__main__":
    main()
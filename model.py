import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel

# ==========================================
# 1. Configuration Class
# ==========================================

class MoEConfig(PretrainedConfig):
    """
    Configuration class for the MoE model.
    Inherits from PretrainedConfig to support HF Trainer integration.
    """
    model_type = "poseidon_moe"

    def __init__(self, 
                 img_size=128,          # Input image resolution (H, W)
                 patch_size=4,          # Patch size for tokenization
                 embed_dim=128,         # Hidden embedding dimension
                 text_dim=768,          # Dimension of text embeddings
                 max_num_channels=256,  # Max number of physical channels supported
                 num_experts=4,         # Total number of experts
                 top_k=2,               # Number of experts activated per token
                 fno_modes=16,          # Number of Fourier modes for FNO expert
                 swin_window_size=8,    # Window size for Window Attention expert
                 swin_num_heads=4,      # Number of attention heads
                 mlp_ratio=4.0,         # Expansion ratio for MLP expert
                 drop_rate=0.0,         # Dropout rate
                 **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.text_dim = text_dim
        self.max_num_channels = max_num_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.fno_modes = fno_modes
        self.swin_window_size = swin_window_size
        self.swin_num_heads = swin_num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        
        # Derived parameters
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

# ==========================================
# 2. Encoder
# ==========================================

class ChannelAwarePatchEncoder(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.shared_patch_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=config.embed_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        self.channel_pos_embed = nn.Embedding(config.max_num_channels, config.embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.embed_dim))
        self.text_proj = nn.Linear(config.text_dim, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
        
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_pos_embed.weight, std=0.02)

    def forward(self, pixel_values, pixel_mask, text_embedding, channel_ids=None):
        B, C, H, W = pixel_values.shape
        N = self.config.num_patches
        
        # Patch Embedding
        x_flat = pixel_values.reshape(B * C, 1, H, W)
        patches = self.shared_patch_embed(x_flat)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.view(B, C, N, self.config.embed_dim)
        
        # Channel Positional Embedding
        if channel_ids is not None:
            ids = channel_ids
        else:
            ids = torch.arange(C, device=pixel_values.device).expand(B, C)
        
        ch_embeds = self.channel_pos_embed(ids)
        patches = patches + ch_embeds.unsqueeze(2)
        
        # Aggregation
        mask_expanded = pixel_mask.view(B, C, 1, 1)
        patches = patches * mask_expanded
        x_agg = patches.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        
        # Spatial Position & Text Context
        x_agg = x_agg + self.spatial_pos_embed
        text_feat = self.text_proj(text_embedding).unsqueeze(1)
        x = x_agg + text_feat
        
        x = self.norm(x)
        return x

# ==========================================
# 3. Experts
# ==========================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        
        # [FIX for NCCL] Store as float32 with last dim=2 (real, imag)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )

    def complex_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        # Intermediate cfloat tensor (safe for local computation)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # [FIX] View float32 params as complex for computation
        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.grid_size = config.grid_size
        self.modes = min(config.fno_modes, self.grid_size // 2)
        self.dim = config.embed_dim
        
        self.conv = SpectralConv2d(self.dim, self.dim, self.modes, self.modes)
        self.w = nn.Conv2d(self.dim, self.dim, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        B, N, D = x.shape
        H = W = self.grid_size
        
        x_grid = x.transpose(1, 2).view(B, D, H, W)
        x1 = self.conv(x_grid)
        x2 = self.w(x_grid)
        out = self.act(x1 + x2)
        
        out = out.flatten(2).transpose(1, 2)
        return out + x 

class MLPExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.drop_rate)
        
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + shortcut

class WindowAttentionExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.dim = config.embed_dim
        self.window_size = config.swin_window_size
        self.num_heads = config.swin_num_heads
        self.grid_size = config.grid_size
        
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, int(self.dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.dim * config.mlp_ratio), self.dim)
        )
        
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        shortcut = x
        B, N, C = x.shape
        H = W = self.grid_size
        
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        x_windows = self.window_partition(x, self.window_size) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 
        
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, N, C)
        
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 4. MoE Processor
# ==========================================

class TopKGating(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.k = config.top_k
        self.gate = nn.Sequential(
            nn.Linear(config.embed_dim + config.text_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.num_experts)
        )
        
    def forward(self, x, text_embedding):
        x_mean = x.mean(dim=1)
        decision_feat = torch.cat([x_mean, text_embedding], dim=1)
        logits = self.gate(decision_feat)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=1)
        top_k_weights = F.softmax(top_k_logits, dim=1)
        return top_k_weights, top_k_indices, logits

class MoEProcessor(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.experts = nn.ModuleList([
            FNOExpert(config),             # Expert 0
            MLPExpert(config),             # Expert 1
            WindowAttentionExpert(config)  # Expert 2
        ])
        
        current_experts = len(self.experts)
        if config.num_experts > current_experts:
            for _ in range(config.num_experts - current_experts):
                self.experts.append(MLPExpert(config))
                
        self.gating = TopKGating(config)
        
    def forward(self, x, text_embedding):
        B, N, D = x.shape
        weights, indices, logits = self.gating(x, text_embedding)
        
        final_output = torch.zeros_like(x)
        
        for k_idx in range(self.config.top_k):
            idx = indices[:, k_idx] 
            w = weights[:, k_idx].view(B, 1, 1)
            for expert_idx, expert in enumerate(self.experts):
                mask = (idx == expert_idx).view(B, 1, 1).float()
                if mask.sum() > 0:
                    expert_out = expert(x)
                    final_output = final_output + w * expert_out * mask
        
        return final_output, logits

# ==========================================
# 5. Decoder
# ==========================================

class QueryBasedDecoder(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.head = nn.Linear(config.embed_dim, self.patch_size**2)
        self.channel_pos_embed = None 

    def forward(self, x, pixel_mask, channel_ids=None):
        B, N, D = x.shape
        C_out = pixel_mask.shape[1]
        
        x_expanded = x.unsqueeze(1).expand(-1, C_out, -1, -1)
        
        if channel_ids is not None:
            ids = channel_ids
        else:
            ids = torch.arange(C_out, device=x.device).expand(B, C_out)
        
        if self.channel_pos_embed is not None:
            ch_embeds = self.channel_pos_embed(ids)
        else:
            raise RuntimeError("channel_pos_embed is not bound! Check model initialization.")
        
        x_query = x_expanded + ch_embeds.unsqueeze(2)
        x_flat = x_query.reshape(-1, D)
        patches_out = self.head(x_flat)
        
        patches_out = patches_out.view(B, C_out, N, self.patch_size, self.patch_size)
        H_grid = W_grid = int(N**0.5)
        patches_out = patches_out.view(B, C_out, H_grid, W_grid, self.patch_size, self.patch_size)
        x_recon = patches_out.permute(0, 1, 2, 4, 3, 5)
        x_recon = x_recon.reshape(B, C_out, H_grid * self.patch_size, W_grid * self.patch_size)
        
        mask = pixel_mask.view(B, C_out, 1, 1)
        x_recon = x_recon * mask
        return x_recon

# ==========================================
# 6. Main Model: PoseidonMoE
# ==========================================

class PoseidonMoE(PreTrainedModel):
    """
    Main Model Class.
    Inherits from PreTrainedModel for full HF integration.
    """
    config_class = MoEConfig

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        
        self.encoder = ChannelAwarePatchEncoder(config)
        self.processor = MoEProcessor(config)
        self.decoder = QueryBasedDecoder(config)
        
        # Share Channel Embeddings
        self.decoder.channel_pos_embed = self.encoder.channel_pos_embed
        
        # Initialize weights via HF method or custom
        self.post_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, 
                pixel_values: torch.Tensor, 
                text_embedding: torch.Tensor, 
                pixel_mask: torch.Tensor, 
                channel_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):
        
        embedding_output = self.encoder(
            pixel_values, 
            pixel_mask, 
            text_embedding, 
            channel_ids=channel_ids
        )
        
        latent_output, gate_logits = self.processor(embedding_output, text_embedding)
        
        output = self.decoder(
            latent_output, 
            pixel_mask, 
            channel_ids=channel_ids
        )
        
        return output, gate_logits

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
    model_type = "poseidon_moe"

    def __init__(self, 
                 img_size=128,          
                 patch_size=4,          
                 # [Upgrade 1] Global Width: 128 -> 256 (4x params for linear layers)
                 embed_dim=256,         
                 text_dim=768,          
                 max_num_channels=256,  
                 num_experts=7,         
                 top_k=2,               
                 
                 # [Upgrade 2] Expert Configurations for ~10M-15M params each
                 
                 # FNO: 256 dim, 8 modes, 2 layers => ~16M params
                 fno_modes=8,           
                 fno_layers=2,          

                 # MLP: 256 dim, ratio 4, 16 layers => ~13M params
                 mlp_layers=16,          
                 
                 # Swin: 256 dim, 12 layers => ~12.6M params (Base-size encoder)
                 swin_layers=12,          
                 
                 # OFormer: 256 dim, 12 layers => ~12.6M params
                 oformer_layers=12,       
                 
                 # KNO: 256 dim, 16 layers => ~12.5M params
                 kno_layers=16,           
                 
                 # WNO: 256 dim, 4 layers => ~9.6M params
                 wno_layers=4,           
                 
                 # U-Net: Scaling follows embed_dim automatically (~11M at dim=256)
                 
                 # Other params
                 swin_window_size=8,    
                 swin_num_heads=8,      # Increased heads for dim 256
                 mlp_ratio=4.0,         
                 drop_rate=0.0,         
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
        self.fno_layers = fno_layers
        self.swin_window_size = swin_window_size
        self.swin_num_heads = swin_num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        
        # Expert Depths
        self.mlp_layers = mlp_layers
        self.swin_layers = swin_layers
        self.oformer_layers = oformer_layers
        self.kno_layers = kno_layers
        self.wno_layers = wno_layers
        
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
        
        x_flat = pixel_values.reshape(B * C, 1, H, W)
        patches = self.shared_patch_embed(x_flat)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.view(B, C, N, self.config.embed_dim)
        
        if channel_ids is not None:
            ids = channel_ids
        else:
            ids = torch.arange(C, device=pixel_values.device).expand(B, C)
        
        ch_embeds = self.channel_pos_embed(ids)
        patches = patches + ch_embeds.unsqueeze(2)
        
        mask_expanded = pixel_mask.view(B, C, 1, 1)
        patches = patches * mask_expanded
        x_agg = patches.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        
        x_agg = x_agg + self.spatial_pos_embed
        text_feat = self.text_proj(text_embedding).unsqueeze(1)
        x = x_agg + text_feat
        
        x = self.norm(x)
        return x

# ==========================================
# 3. Experts
# ==========================================

# --- FNO Expert (Refactored to Block Stacking) ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        
        # Real and Imaginary parts split for NCCL compatibility
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
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        w1 = torch.view_as_complex(self.weights1)
        w2 = torch.view_as_complex(self.weights2)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock(nn.Module):
    """
    Standard FNO Block: SpectralConv -> Skip -> Act
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        self.modes = min(config.fno_modes, config.grid_size // 2)
        
        self.conv = SpectralConv2d(self.dim, self.dim, self.modes, self.modes)
        self.w = nn.Conv2d(self.dim, self.dim, 1)
        self.act = nn.GELU()
        # Add norm for stability in deep FNOs
        self.norm = nn.LayerNorm(self.dim) 

    def forward(self, x_grid):
        # x_grid: [B, D, H, W]
        x1 = self.conv(x_grid)
        x2 = self.w(x_grid)
        out = self.act(x1 + x2)
        return out

class FNOExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.grid_size = config.grid_size
        self.dim = config.embed_dim
        
        self.layers = nn.ModuleList([
            FNOBlock(config) for _ in range(config.fno_layers)
        ])
        
    def forward(self, x):
        B, N, D = x.shape
        H = W = self.grid_size
        
        # [B, N, D] -> [B, D, H, W]
        x_grid = x.transpose(1, 2).view(B, D, H, W)
        
        shortcut = x_grid
        for layer in self.layers:
            x_grid = layer(x_grid)
        
        # Residual connection over the whole expert
        x_grid = x_grid + shortcut
        
        # [B, D, H, W] -> [B, N, D]
        out = x_grid.flatten(2).transpose(1, 2)
        return out

# --- MLP Expert ---
class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + shortcut

class MLPExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MLPBlock(config) for _ in range(config.mlp_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Swin Expert ---
class WindowAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        self.window_size = config.swin_window_size
        self.num_heads = config.swin_num_heads
        self.grid_size = config.grid_size
        self.mlp_ratio = config.mlp_ratio
        
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, int(self.dim * self.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(self.dim * self.mlp_ratio), self.dim)
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
        
        x_norm = self.norm1(x)
        x_norm = x_norm.view(B, H, W, C)
        
        x_windows = self.window_partition(x_norm, self.window_size) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 
        
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x_attn = self.window_reverse(attn_windows, self.window_size, H, W)
        x_attn = x_attn.view(B, N, C)
        
        x = shortcut + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

class WindowAttentionExpert(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            WindowAttnBlock(config) for _ in range(config.swin_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- OFormer Expert ---
class GalerkinBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.embed_dim
        num_heads = 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.ln_k = nn.LayerNorm(head_dim)
        self.ln_v = nn.LayerNorm(head_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * config.mlp_ratio), dim)
        )

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape
        
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        k = self.ln_k(k)
        v = self.ln_v(v)
        k_t = k.transpose(-2, -1)
        
        context = torch.matmul(k_t, v) 
        attn_out = torch.matmul(q, context)
        attn_out = attn_out * self.scale
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        
        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class OFormerExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            GalerkinBlock(config) for _ in range(config.oformer_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- U-Net Expert (Scaled Width) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels), # GN groups adjusted for larger width
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetExpert(nn.Module):
    """
    Standard U-Net with 1 downsample stage. 
    Widths: dim -> dim*2 -> dim. 
    At dim=256, params approx 11M.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        
        self.inc = DoubleConv(self.dim, self.dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.dim, self.dim * 2))
        
        self.bot = DoubleConv(self.dim * 2, self.dim * 2)
        
        self.up1 = nn.ConvTranspose2d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.outc = DoubleConv(self.dim * 2, self.dim)
        
        self.final = nn.Conv2d(self.dim, self.dim, kernel_size=1)

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N**0.5)
        x_img = x.transpose(1, 2).view(B, D, H, W)
        
        x1 = self.inc(x_img)         
        x2 = self.down1(x1)          
        x3 = self.bot(x2)            
        x_up = self.up1(x3)          
        
        x_cat = torch.cat([x1, x_up], dim=1)
        x_out = self.outc(x_cat)
        x_out = self.final(x_out)
        
        x_out = x_out.flatten(2).transpose(1, 2)
        return x + x_out 

# --- KNO Expert ---
class KNOBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        self.lift_dim = self.dim * 2 
        self.norm = nn.LayerNorm(self.dim)
        
        self.lift = nn.Linear(self.dim, self.lift_dim)
        self.dynamics = nn.Linear(self.lift_dim, self.lift_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(self.lift_dim, self.dim)

    def forward(self, x):
        shortcut = x
        x_in = self.norm(x)
        x = self.lift(x_in)
        x = self.act(x)
        x = self.dynamics(x) 
        x = self.act(x)
        x = self.proj(x)
        return shortcut + x

class KNOExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            KNOBlock(config) for _ in range(config.kno_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- WNO Expert ---
class HaarWavelet2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer('filters', torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],      # LL
            [[0.5, -0.5], [0.5, -0.5]],    # LH 
            [[0.5, 0.5], [-0.5, -0.5]],    # HL 
            [[0.5, -0.5], [-0.5, 0.5]]     # HH 
        ]).unsqueeze(1).repeat(in_channels, 1, 1, 1))
        
        self.register_buffer('inv_filters', torch.tensor([
            [[0.5, 0.5], [0.5, 0.5]],      
            [[0.5, -0.5], [0.5, -0.5]],    
            [[0.5, 0.5], [-0.5, -0.5]],    
            [[0.5, -0.5], [-0.5, 0.5]]     
        ]).unsqueeze(1).repeat(in_channels, 1, 1, 1)) 

    def forward(self, x):
        return F.conv2d(x, self.filters, stride=2, groups=self.in_channels)

    def inverse(self, x):
        return F.conv_transpose2d(x, self.inv_filters, stride=2, groups=self.in_channels)

class WNOBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        self.wavelet = HaarWavelet2d(self.dim)
        
        # WNO Block logic: DWT -> Conv(mix bands) -> IDWT
        # Input channels to Conv is 4 * Dim (LL, LH, HL, HH)
        self.conv = nn.Sequential(
            nn.Conv2d(self.dim * 4, self.dim * 4, kernel_size=3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv2d(self.dim * 4, self.dim * 4, kernel_size=1)
        )
        self.mixing = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        shortcut = x
        x_in = self.norm(x)
        
        B, N, D = x_in.shape
        H = W = int(N**0.5)
        x_img = x_in.transpose(1, 2).view(B, D, H, W)
        
        x_freq = self.wavelet(x_img)
        x_processed = self.conv(x_freq)
        x_recon = self.wavelet.inverse(x_processed)
        
        x_out = x_recon.flatten(2).transpose(1, 2)
        return shortcut + self.mixing(x_out)

class WNOExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            WNOBlock(config) for _ in range(config.wno_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# 5. MoE Processor
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
            FNOExpert(config),             # Expert 0: Fourier
            MLPExpert(config),             # Expert 1: MLP (Deep)
            WindowAttentionExpert(config), # Expert 2: Swin (Deep)
            OFormerExpert(config),         # Expert 3: OFormer (Deep)
            UNetExpert(config),            # Expert 4: U-Net (Wide)
            KNOExpert(config),             # Expert 5: KNO (Deep)
            WNOExpert(config)              # Expert 6: WNO (Deep)
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
                # Run expert regardless of mask (for DDP stability)
                expert_out = expert(x)
                final_output = final_output + w * expert_out * mask
        
        return final_output, logits

# ==========================================
# 6. Decoder
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
# 7. Main Model: PoseidonMoE
# ==========================================

class PoseidonMoE(PreTrainedModel):
    config_class = MoEConfig

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        
        self.encoder = ChannelAwarePatchEncoder(config)
        self.processor = MoEProcessor(config)
        self.decoder = QueryBasedDecoder(config)
        
        self.decoder.channel_pos_embed = self.encoder.channel_pos_embed
        
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

# ==========================================
# 1. 配置类 (Configuration)
# ==========================================

class MoEConfig:
    """
    MoE 模型配置类，集中管理所有超参数
    """
    def __init__(self, 
                 img_size=128,          # 输入图像分辨率
                 patch_size=4,          # Patch 分块大小
                 embed_dim=128,         # 隐层特征维度
                 text_dim=768,          # 文本 Embedding 维度 (Roberta-base)
                 max_num_channels=256,  # 支持的最大物理通道数 (用于位置编码库大小)
                 num_experts=4,         # 专家总数
                 top_k=2,               # 每次激活的专家数
                 fno_modes=16,          # FNO 模态数 (cutoff frequency)
                 swin_window_size=8,    # Swin Attention 窗口大小
                 swin_num_heads=4,      # Attention 头数
                 mlp_ratio=4.0,         # MLP 膨胀比
                 drop_rate=0.0,         # Dropout 率
                 **kwargs):
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
        
        # 计算网格参数
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

# ==========================================
# 2. 编码器 (Shared Encoder with Positional Channel Embedding)
# ==========================================

class ChannelAwarePatchEncoder(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # 1. 共享 Patch Embedding (处理单个通道)
        # Input: [B*C, 1, H, W] -> Output: [B*C, Dim, H/P, W/P]
        self.shared_patch_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=config.embed_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        # 2. 通道位置编码 (Channel Positional Encoding)
        # 用于区分这是第几个通道 (Channel 0, Channel 1...)
        self.channel_pos_embed = nn.Embedding(config.max_num_channels, config.embed_dim)
        
        # 3. 空间位置编码 (Spatial Positional Encoding)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.embed_dim))
        
        # 4. 文本投影层 (将 768维 文本特征映射到隐层维度)
        self.text_proj = nn.Linear(config.text_dim, config.embed_dim)
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_pos_embed.weight, std=0.02)

    def forward(self, pixel_values, pixel_mask, text_embedding):
        """
        Args:
            pixel_values: [B, C, H, W]
            pixel_mask:   [B, C] (1=Valid, 0=Padding)
            text_embedding: [B, Text_Dim] (包含方程+变量描述的文本编码)
        Returns:
            x: [B, N, Dim]
        """
        B, C, H, W = pixel_values.shape
        N = self.config.num_patches
        
        # --- Step 1: Patch Embedding (Shared) ---
        # Reshape: [B, C, H, W] -> [B*C, 1, H, W]
        x_flat = pixel_values.reshape(B * C, 1, H, W)
        
        # Conv: [B*C, Dim, grid, grid]
        patches = self.shared_patch_embed(x_flat)
        
        # Flatten: [B*C, Dim, N] -> [B*C, N, Dim]
        patches = patches.flatten(2).transpose(1, 2)
        
        # Restore Shape: [B, C, N, Dim]
        patches = patches.view(B, C, N, self.config.embed_dim)
        
        # --- Step 2: Add Channel Positional Embedding ---
        # 生成通道索引: [0, 1, ..., C-1]
        # ids: [B, C]
        ids = torch.arange(C, device=pixel_values.device).expand(B, C)
        
        # ch_embeds: [B, C, Dim]
        ch_embeds = self.channel_pos_embed(ids)
        
        # Broadcast add: [B, C, N, Dim] + [B, C, 1, Dim]
        patches = patches + ch_embeds.unsqueeze(2)
        
        # --- Step 3: Aggregation (Weighted Average by Mask) ---
        # mask: [B, C] -> [B, C, 1, 1]
        mask_expanded = pixel_mask.view(B, C, 1, 1)
        
        # sum(patches * mask) / sum(mask)
        patches = patches * mask_expanded
        x_agg = patches.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6) # [B, N, Dim]
        
        # --- Step 4: Add Spatial Position Embedding ---
        x_agg = x_agg + self.spatial_pos_embed
        
        # --- Step 5: Inject Text Context ---
        # text: [B, Text_Dim] -> [B, 1, Dim]
        text_feat = self.text_proj(text_embedding).unsqueeze(1)
        x = x_agg + text_feat
        
        x = self.norm(x)
        return x

# ==========================================
# 3. 专家模型 (Experts)
# ==========================================

# --- Expert A: FNO (Fourier Neural Operator) ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def complex_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
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
        # x: [B, N, D]
        B, N, D = x.shape
        H = W = self.grid_size
        
        # Reshape to Grid: [B, D, H, W]
        x_grid = x.transpose(1, 2).view(B, D, H, W)
        
        # FNO Ops
        x1 = self.conv(x_grid)
        x2 = self.w(x_grid)
        out = self.act(x1 + x2)
        
        # Flatten back: [B, N, D]
        out = out.flatten(2).transpose(1, 2)
        return out + x # Residual

# --- Expert B: Gated MLP (DeepONet Style) ---
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
        # x: [B, N, D]
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + shortcut

# --- Expert C: Swin V2 Block (Window Attention) ---
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
        
        # Partition
        x_windows = self.window_partition(x, self.window_size) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 
        
        # Attention
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        
        # Reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, N, C)
        
        # FFN
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 4. MoE 处理器 (Top-K Gating)
# ==========================================

class TopKGating(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.k = config.top_k
        # 门控输入: 图像全局特征 + 文本特征
        self.gate = nn.Sequential(
            nn.Linear(config.embed_dim + config.text_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.num_experts)
        )
        
    def forward(self, x, text_embedding):
        # x: [B, N, D] -> Global Mean [B, D]
        x_mean = x.mean(dim=1)
        
        # Concat: [B, D + Text_Dim]
        decision_feat = torch.cat([x_mean, text_embedding], dim=1)
        
        # Logits: [B, Num_Experts]
        logits = self.gate(decision_feat)
        
        # Top-K
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=1)
        top_k_weights = F.softmax(top_k_logits, dim=1)
        
        return top_k_weights, top_k_indices

class MoEProcessor(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # 定义专家列表
        self.experts = nn.ModuleList([
            FNOExpert(config),             # Expert 0
            MLPExpert(config),             # Expert 1
            WindowAttentionExpert(config)  # Expert 2
        ])
        
        # 填充更多专家以满足 num_experts
        current_experts = len(self.experts)
        if config.num_experts > current_experts:
            for _ in range(config.num_experts - current_experts):
                self.experts.append(MLPExpert(config))
                
        self.gating = TopKGating(config)
        
    def forward(self, x, text_embedding):
        B, N, D = x.shape
        weights, indices = self.gating(x, text_embedding)
        
        final_output = torch.zeros_like(x)
        
        # 稀疏执行循环
        for k_idx in range(self.config.top_k):
            # 获取第 k 个选择的专家索引和权重
            idx = indices[:, k_idx] # [B]
            w = weights[:, k_idx].view(B, 1, 1) # [B, 1, 1]
            
            for expert_idx, expert in enumerate(self.experts):
                # 找出 batch 中命中当前专家的样本
                mask = (idx == expert_idx).view(B, 1, 1).float()
                
                if mask.sum() > 0:
                    expert_out = expert(x)
                    final_output = final_output + w * expert_out * mask
                    
        return final_output, weights

# ==========================================
# 5. 解码器 (Query-Based Decoder)
# ==========================================

class QueryBasedDecoder(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        
        # 共享投影头: Dim -> Patch_Size^2
        self.head = nn.Linear(config.embed_dim, self.patch_size**2)
        
        # 复用 Encoder 的通道位置编码 (将在主模型中绑定)
        self.channel_pos_embed = None 

    def forward(self, x, pixel_mask):
        """
        x: [B, N, D]
        pixel_mask: [B, C_out]
        """
        B, N, D = x.shape
        C_out = pixel_mask.shape[1]
        
        # --- Step 1: 准备 Query ---
        # 扩展 Latent: [B, N, D] -> [B, C_out, N, D]
        x_expanded = x.unsqueeze(1).expand(-1, C_out, -1, -1)
        
        # 生成通道 ID: [0, 1, ..., C_out-1]
        ids = torch.arange(C_out, device=x.device).expand(B, C_out)
        
        # 获取位置编码: [B, C_out, D]
        ch_embeds = self.channel_pos_embed(ids)
        
        # 将位置编码作为 Query 加到 Latent 上
        # [B, C_out, N, D] + [B, C_out, 1, D]
        x_query = x_expanded + ch_embeds.unsqueeze(2)
        
        # --- Step 2: 共享解码 ---
        # 合并维度进行并行投影: [B*C_out*N, D]
        x_flat = x_query.reshape(-1, D)
        
        # Projection: [..., P*P]
        patches_out = self.head(x_flat)
        
        # --- Step 3: Reshape 回图像 ---
        # [B, C_out, N, P*P]
        patches_out = patches_out.view(B, C_out, N, self.patch_size, self.patch_size)
        
        H_grid = W_grid = int(N**0.5)
        
        # Grid Reshape: [B, C, H_grid, W_grid, P, P]
        patches_out = patches_out.view(B, C_out, H_grid, W_grid, self.patch_size, self.patch_size)
        
        # Permute: [B, C, H_grid, P, W_grid, P]
        x_recon = patches_out.permute(0, 1, 2, 4, 3, 5)
        
        # Final: [B, C, H, W]
        x_recon = x_recon.reshape(B, C_out, H_grid * self.patch_size, W_grid * self.patch_size)
        
        # --- Step 4: Masking ---
        # 仅保留有效通道
        mask = pixel_mask.view(B, C_out, 1, 1)
        x_recon = x_recon * mask
        
        return x_recon

# ==========================================
# 6. 主模型 (PoseidonMoE)
# ==========================================

class PoseidonMoE(nn.Module):
    def __init__(self, config: Optional[MoEConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = MoEConfig(**kwargs)
        self.config = config
        
        # 实例化组件
        self.encoder = ChannelAwarePatchEncoder(config)
        self.processor = MoEProcessor(config)
        self.decoder = QueryBasedDecoder(config)
        
        # 【关键】共享通道位置编码权重
        # Encoder 学习到的 "Channel 0" 的特征，Decoder 也用同样的特征去查询 "Channel 0"
        self.decoder.channel_pos_embed = self.encoder.channel_pos_embed
        
        self.apply(self._init_weights)

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
                **kwargs):
        """
        Args:
            pixel_values: [B, C, H, W]
            text_embedding: [B, 768] (包含方程描述+变量名称列表的文本)
            pixel_mask: [B, C]
        """
        # 1. 编码器: [B, C, H, W] -> [B, N, D]
        # 内部使用位置编码区分通道，并加权融合
        embedding_output = self.encoder(pixel_values, pixel_mask, text_embedding)
        
        # 2. MoE 处理器: [B, N, D] -> [B, N, D]
        # Top-K 专家混合处理
        latent_output, _ = self.processor(embedding_output, text_embedding)
        
        # 3. 解码器: [B, N, D] -> [B, C, H, W]
        # 使用位置编码查询，还原出多通道
        output = self.decoder(latent_output, pixel_mask)
        
        return output

# ==========================================
# 7. 维度测试
# ==========================================
if __name__ == "__main__":
    # 配置
    config = MoEConfig(
        img_size=128, 
        patch_size=4, 
        embed_dim=64,
        num_experts=3,
        top_k=2
    )
    model = PoseidonMoE(config)
    
    # 模拟输入 (Batch=2, Channels=3)
    B, C, H, W = 2, 5, 128, 128
    pixel_values = torch.randn(B, C, H, W)
    pixel_mask = torch.ones(B, C)
    text_embedding = torch.randn(B, 768)
    
    print("Start Forward Pass...")
    out = model(pixel_values, text_embedding, pixel_mask)
    
    print(f"Input: {pixel_values.shape}")
    print(f"Output: {out.shape}")
    
    assert out.shape == pixel_values.shape
    print("Test Passed!")

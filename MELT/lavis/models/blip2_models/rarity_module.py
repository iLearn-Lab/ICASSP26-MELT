import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import rtuple


# attention bloc
class CrossAttnResidual(nn.Module):
    def __init__(self, dim, num_heads=8, topk=5):
        super().__init__()
        self.topk = topk
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.stats = EMAStats(dim, momentum=0.9)

    def forward(self, text_cls, image_token):
        query = self.q_proj(text_cls).unsqueeze(1)          # [B, 1, D]
        key = val = self.kv_proj(image_token)               # [B, S, D]
        out, w = self.cross_attn(query, key, val)           # out:[B,1,D], w:[B,1,S]
        weights = w.squeeze(1)                              # [B, S]


        topk_w, idx = torch.topk(weights, k=self.topk, dim=-1)  # [B, topk]


        topk_w = F.softmax(topk_w, dim=-1)  # [B, topk]  ∑=1


        selected = image_token[torch.arange(image_token.size(0)).unsqueeze(1), idx]  # [B, topk, D]


        selected_pooled = torch.sum(topk_w.unsqueeze(-1) * selected, dim=1)  # [B, D]


        # ===== Mahalanobis 残差能量 Residual vector=====
        residual = text_cls - selected_pooled  # [B, D]
        self.stats.update(residual)  # 仅训练期更新
        cov_inv = self.stats.inv()  # [D, D]

        mah_sq = torch.sum(residual @ cov_inv * residual, dim=1)  # [B]
        residual_energy = torch.sqrt(mah_sq + 1e-6)  # [B]


        topk_weights = weights.gather(1, idx).mean(dim=1)  # [B]  ∈[0,1]

        # global_sim = (text_cls * selected_pooled).sum(dim=-1)  # [B]  ∈[-1,1]
        global_sim = F.cosine_similarity(text_cls, selected_pooled, dim=1)
        global_sim = torch.sigmoid(global_sim)  # 归一化到 [0,1]

        rarity_score = residual_energy  * (1 - global_sim)
        return rarity_score, w, idx, selected_pooled



#nterest for calculating the rarity score
class EMAStats(nn.Module):
    """
    在线 EMA 估计均值 & 协方差，并提供逆矩阵用于马氏距离
    """
    def __init__(self, dim, momentum=0.9, eps=1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('mean', torch.zeros(dim))          # μ
        self.register_buffer('cov', torch.eye(dim))             # Σ
        self.register_buffer('cov_inv', torch.eye(dim))         # Σ⁻¹
        self.register_buffer('count', torch.tensor(0.))         # 用于冷启动保护
        self.dim = dim


    @torch.no_grad()
    def update(self, x):
        """x: [B, D] 的残差或原始向量"""
        if self.training:
            batch_mean = x.mean(0)
            batch_cov  = torch.mm((x - batch_mean).t(), x - batch_mean) / x.size(0)
            # EMA 更新
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.cov  = self.momentum * self.cov  + (1 - self.momentum) * batch_cov
            eye_matrix = torch.eye(self.dim, device=self.cov.device, dtype=self.cov.dtype)
            try:
                U = torch.linalg.cholesky(self.cov + self.eps * eye_matrix)
                self.cov_inv = torch.cholesky_inverse(U)
            except RuntimeError:
                self.cov_inv = torch.linalg.pinv(self.cov + self.eps * eye_matrix)
            self.count += 1

    def inv(self):
        """返回当前 EMA 估计的协方差逆矩阵"""
        return self.cov_inv



# hreshold for rare sample classification
class RarityJudge(nn.Module):
    def __init__(self, momentum=0.9, init_thresh=2.2):
        super().__init__()
        self.momentum = momentum
        # self.z_thresh = z_thresh
        self.register_buffer('mean', torch.tensor(0.))
        self.register_buffer('var', torch.tensor(1.))
        self.raw_thresh = nn.Parameter(torch.tensor(init_thresh))


    @property
    def z_thresh(self):
        # return F.softplus(self.raw_thresh).clamp(0.5, 3.5)  # >0
        return self.raw_thresh


    @torch.no_grad()
    def forward(self, energy):
        # EMA 更新
        self.mean = self.momentum * self.mean + (1 - self.momentum) * energy.mean()
        self.var  = self.momentum * self.var  + (1 - self.momentum) * energy.var()
        z = (energy - self.mean) / (self.var.sqrt() + 1e-6)
        # thresh=self.z_thresh
        # print("z_thresh",thresh)

        return z > self.z_thresh


#  modification to the image tokens
class TokenCorrector(nn.Module):
    def __init__(self, dim, init_strength=0.8):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(float(init_strength)))

    def forward(self, image_token, text_cls, topk_idx, selected_pooled, is_rare):
        delta = F.normalize(text_cls, dim=-1) - F.normalize(selected_pooled, dim=-1)
        delta = delta * self.strength * is_rare.float().unsqueeze(1)   # [B, D] * [B, 1]
        strength=self.strength
        # print("strength",strength)
        updated = image_token.clone()
        B, topk = topk_idx.shape
        for b in range(B):
            if is_rare[b]:
                updated[b, topk_idx[b]] += delta[b].unsqueeze(0).expand(topk, -1)
        return image_token




class RarityFix(nn.Module):
    def __init__(self, dim, num_heads=8, topk=5, strength=0.2):
        super().__init__()
        self.cross = CrossAttnResidual(dim, num_heads, topk)
        self.judge = RarityJudge(momentum=0.9, init_thresh=2.2)
        self.corrector = TokenCorrector(dim, strength)

    def forward(self, text_cls, image_token):
        """纯修正，无损失返回"""

        # energy is fitting score
        energy, _, idx, pooled = self.cross(text_cls, image_token)
        is_rare = self.judge(energy)
        if self.training:
            image_token = self.corrector(image_token, text_cls, idx, pooled, is_rare)
        # return energy,is_rare
        return image_token
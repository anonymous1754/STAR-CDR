# Model.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# 一些基础模块
# ------------------------

class MLP(nn.Module):
    """通用多层感知机模块：dims = [in_dim, h1, h2, ..., out_dim]"""
    def __init__(self,
                 dims: List[int],
                 activation: str = "relu",
                 dropout: float = 0.0,
                 last_activation: bool = False):
        super().__init__()
        layers = []
        act_layer = nn.ReLU if activation == "relu" else nn.GELU
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or last_activation:
                layers.append(act_layer())
            if dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ScaledDotProductAttention(nn.Module):
    """简化版 Cross-Attention：单头，Q: [B, Lq, D], K/V: [B, Lk, D]"""
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = d_model ** 0.5

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        # q: [B, Lq, D], k,v: [B, Lk, D]
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # [B, Lq, Lk]
        if mask is not None:
            # mask: [B, 1, Lk]，1 为可见，0 为不可见
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Lq, Lk]
        out = torch.matmul(attn_weights, v)  # [B, Lq, D]
        return out, attn_weights


class CrossNetV2(nn.Module):
    """
    DCNv2 风格的 Cross Network
    使用低秩矩阵分解来提高效率
    """
    def __init__(self, input_dim: int, num_layers: int = 2, low_rank: Optional[int] = None):
        super().__init__()
        self.num_layers = num_layers
        if low_rank is None:
            low_rank = input_dim // 4
        
        self.cross_layers = nn.ModuleList()
        for _ in range(num_layers):
            # 使用低秩矩阵分解：W = U * V^T
            self.cross_layers.append(nn.ModuleDict({
                'U': nn.Linear(input_dim, low_rank, bias=False),
                'V': nn.Linear(low_rank, input_dim, bias=False),
                'bias': nn.ParameterDict({'b': nn.Parameter(torch.zeros(input_dim))})
            }))
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        x0: [B, D] 初始输入
        返回: [B, D] 经过 cross network 后的输出
        """
        x_i = x0  # [B, D]
        
        for layer in self.cross_layers:
            # x_{i+1} = x_0 * (W * x_i + b) + x_i
            # 使用低秩分解: W = U * V^T
            x_i_transformed = layer['V'](layer['U'](x_i))  # [B, D]
            x_i_transformed = x_i_transformed + layer['bias']['b']  # [B, D]
            x_i = x0 * x_i_transformed + x_i  # [B, D]
        
        return x_i


# ------------------------
# CDI Extractor
# ------------------------

class CDIExtractor(nn.Module):
    """
    对应伪代码 Algorithm 1：CDI Extractor
    输入：
        u: [B, Du] 用户向量
        H_intra: Dict[domain_id, Tensor]，每个 value: [B, L_d, Dh]
        H_share: [B, Ls, Dh]
    输出：
        cdir_dict: {d -> [B, Dcdir]}
        contrastive_loss: 标量
    """

    def __init__(self,
                 num_domains: int,
                 u_dim: int,
                 ctx_dim: int,
                 hidden_dim: int,
                 cdir_dim: int,
                 qkv_dim: int):
        super().__init__()
        self.num_domains = num_domains
        self.hidden_dim = hidden_dim
        self.cdir_dim = cdir_dim
        self.qkv_dim = qkv_dim

        # Q 先把 u 和 C^(d,s) concat 再投影
        self.w_q = nn.Linear(u_dim + ctx_dim, qkv_dim)
        # K,V 对 intra / share 序列分别使用相同投影
        self.w_k_intra = nn.Linear(hidden_dim, qkv_dim)
        self.w_v_intra = nn.Linear(hidden_dim, qkv_dim)

        self.w_k_share = nn.Linear(hidden_dim, qkv_dim)
        self.w_v_share = nn.Linear(hidden_dim, qkv_dim)

        self.mca_intra = ScaledDotProductAttention(qkv_dim)
        self.mca_share = ScaledDotProductAttention(qkv_dim)

        # FFN 把两路拼接压缩成 CDIR
        self.ffn_cdir = MLP([2 * qkv_dim, cdir_dim])

    def forward(self,
                u: torch.Tensor,
                H_intra: Dict[int, torch.Tensor],
                H_share: torch.Tensor,
                ctx: Optional[Dict[int, torch.Tensor]] = None
                ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        ctx: 可选，C^(d,s)，若为 None，则用 H_intra^d 的 mean 代替
        """
        batch_size = u.size(0)
        device = u.device

        # 预处理 shared 序列
        K_share = self.w_k_share(H_share)       # [B, Ls, Dq]
        V_share = self.w_v_share(H_share)       # [B, Ls, Dq]

        cdir_dict: Dict[int, torch.Tensor] = {}
        h_intra_list = []   # 用于对比学习
        h_share_list = []   # 用于对比学习

        for d in range(self.num_domains):
            h_d = H_intra[d]   # [B, Ld, Dh]

            if ctx is not None and d in ctx:
                c_d = ctx[d]   # [B, Dc]
            else:
                # 简单用平均池化作为 C^(d,s)
                c_d = h_d.mean(dim=1)  # [B, Dh]，可以用 Linear 降维到 ctx_dim，在初始化时统一
            # 若 ctx_dim != hidden_dim，请在外部保证 ctx 的维度正确

            # 构造 Q^d
            q_input = torch.cat([u, c_d], dim=-1)  # [B, Du+Dc]
            Q_d = self.w_q(q_input).unsqueeze(1)   # [B,1,Dq]

            # intra-domain cross attention
            K_intra = self.w_k_intra(h_d)  # [B, Ld, Dq]
            V_intra = self.w_v_intra(h_d)
            h_intra, _ = self.mca_intra(Q_d, K_intra, V_intra)  # [B,1,Dq]
            h_intra = h_intra.squeeze(1)                        # [B,Dq]

            # domain-shared cross attention
            h_share, _ = self.mca_share(Q_d, K_share, V_share)  # [B,1,Dq]
            h_share = h_share.squeeze(1)                        # [B,Dq]

            # fuse to CDIR^d
            h_cat = torch.cat([h_intra, h_share], dim=-1)  # [B,2Dq]
            z_d = self.ffn_cdir(h_cat)                     # [B,Dcdir]

            cdir_dict[d] = z_d
            h_intra_list.append(h_intra)
            h_share_list.append(h_share)

        # contrastive loss L_con
        # 按照伪代码：对每个 domain 的 h_intra 与自身 h_share 拉近，与其他 domain 的 h_intra 推远
        # 简化实现：每个样本维度上分别算
        h_intra_stacked = torch.stack(h_intra_list, dim=1)  # [B, D_num, Dq]
        h_share_stacked = torch.stack(h_share_list, dim=1)  # [B, D_num, Dq]

        # sim(a,b) = cosine similarity
        def sim(a, b):
            # a,b: [B, D_num, Dq]
            a_norm = F.normalize(a, dim=-1)
            b_norm = F.normalize(b, dim=-1)
            return (a_norm * b_norm).sum(dim=-1)  # [B, D_num]

        pos_sim = sim(h_intra_stacked, h_share_stacked)  # [B, D_num]

        # neg: h_intra^d 与所有 h_intra^{d'}
        # [B, D_num, D_num]
        h_intra_i = h_intra_stacked.unsqueeze(2)  # [B, D_num,1,Dq]
        h_intra_j = h_intra_stacked.unsqueeze(1)  # [B,1,D_num,Dq]
        a_norm = F.normalize(h_intra_i, dim=-1)
        b_norm = F.normalize(h_intra_j, dim=-1)
        all_sim = (a_norm * b_norm).sum(dim=-1)  # [B,D_num,D_num]

        # diag 是与自身的相似度
        # softmax 分母
        exp_all = torch.exp(all_sim)  # [B,D_num,D_num]
        denom = exp_all.sum(dim=-1)   # [B,D_num]

        num = torch.exp(pos_sim)      # [B,D_num]
        log_prob = torch.log(num / (denom + 1e-8) + 1e-8)  # [B,D_num]
        L_con = -log_prob.mean()

        return cdir_dict, L_con


# ------------------------
# CDI Injector: Gate & Norm
# ------------------------

class CDIRInjector_Gate(nn.Module):
    """
    用 CDIR 生成 gate，支持 domain / scenario / task。
    输入：z_d: [B, Dcdir]
    输出：gate: [B, num_experts]，Sigmoid 后可作为 soft gate 或 mask。
    """
    def __init__(self,
                 cdir_dim: int,
                 num_experts: int,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = cdir_dim
        self.net = MLP([cdir_dim, hidden_dim, num_experts], last_activation=False)

    def forward(self, z: torch.Tensor):
        gate_logits = self.net(z)          # [B, num_experts]
        gate = torch.sigmoid(gate_logits)  # [B, num_experts]
        return gate


class CDIRInjector_Norm(nn.Module):
    """
    AdaNorm 风格，按 domain 做 masked 归一化 + domain 特定的 gamma / beta 注入。
    gamma,beta 由 CDIR 生成。
    """
    def __init__(self,
                 hidden_dim: int,
                 cdir_dim: int,
                 num_domains: int,
                 eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.eps = eps

        # 先把 CDIR 投影到 gamma,beta
        self.to_gamma_beta = nn.Linear(cdir_dim, 2 * hidden_dim)

    def forward(self,
                h: torch.Tensor,
                cdir_dict: Dict[int, torch.Tensor],
                domain_ids: torch.Tensor):
        """
        h: [B, Dh]
        cdir_dict: {d -> [B,Dcdir]}，同一个 batch 中不同 domain 的样本会用到不同的 CDIR
        domain_ids: [B]，取值 0~num_domains-1
        """
        B, Dh = h.shape
        device = h.device
        out = torch.zeros_like(h)

        for d in range(self.num_domains):
            mask = (domain_ids == d)  # [B]
            if not mask.any():
                continue

            idx = mask.nonzero(as_tuple=False).squeeze(1)  # [Bd]
            h_d = h[idx]                                   # [Bd, Dh]
            z_d = cdir_dict[d][idx]                        # [Bd, Dcdir]

            # 计算 gamma,beta
            gb = self.to_gamma_beta(z_d)                   # [Bd, 2Dh]
            gamma, beta = gb.chunk(2, dim=-1)              # [Bd, Dh]

            # 按 domain 做归一化
            mean = h_d.mean(dim=0, keepdim=True)           # [1,Dh]
            var = h_d.var(dim=0, unbiased=False, keepdim=True)
            h_norm = (h_d - mean) / torch.sqrt(var + self.eps)  # [Bd,Dh]

            # AdaNorm 风格注入，这里简单做 gamma * h_norm + beta + residual
            h_out = gamma * h_norm + beta + h_d
            out[idx] = h_out

        return out


# ------------------------
# Expert Modules
# ------------------------

class MoEExperts(nn.Module):
    """
    一个通用的 MoE 模块：
    - experts: N 个相同结构的 MLP
    - gate: [B, N] in [0,1]
    - 输入 x: [B, Din]，输出 [B, Dout]
    """
    def __init__(self,
                 num_experts: int,
                 in_dim: int,
                 expert_hidden_dims: List[int]):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList()
        dims = [in_dim] + expert_hidden_dims
        for _ in range(num_experts):
            self.experts.append(MLP(dims, last_activation=False))

    def forward(self,
                x: torch.Tensor,
                gate: torch.Tensor):
        """
        x: [B, Din]
        gate: [B, num_experts]，通常来自 CDIRInjector_Gate 的 Sigmoid/Softmax 输出
        """
        B, _ = x.size()
        expert_outs = []
        for e in self.experts:
            expert_outs.append(e(x))  # 每个: [B,Dout]
        # [N,B,Dout] -> [B,N,Dout]
        expert_stack = torch.stack(expert_outs, dim=1)
        # gate: [B,N] -> [B,N,1]
        gate = gate.unsqueeze(-1)
        out = (expert_stack * gate).sum(dim=1)  # [B,Dout]
        return out


# ------------------------
# 主模型
# ------------------------

class CDIRecModel(nn.Module):
    """
    整体模型骨架：
    - 输入：all_features: Dict[str, Tensor]
    - 输出：loss, click_prob, play_prob, pay_prob
    """

    def __init__(self,
                 num_domains: int = 5,
                 num_scenarios: int = 3,
                 num_tasks: int = 5,
                 embed_dim: int = 128,
                 expert_hidden_dims: Optional[List[int]] = None,
                 tower_hidden_dims: Optional[List[int]] = None,
                 cdir_dim: int = 64,
                 qkv_dim: int = 64,
                 num_domain_experts: int = 4,
                 num_scenario_experts: int = 4,
                 num_task_experts: int = 4,
                 num_image_experts: int = 4,
                 num_text_experts: int = 4,
                 image_dim: int = 64,
                 text_dim: int = 64,
                 important_features: Optional[List[str]] = None,
                 categorical_features: Optional[Dict[str, int]] = None,
                 categorical_embed_dim: int = 16,
                 continuous_features_dim: Optional[Dict[str, int]] = None,
                 sequence_features: Optional[List[str]] = None,
                 sequence_pooling: str = "mean"):
        """
        Args:
            categorical_features: Dict[特征名, 字典大小]，例如 {"user_id": 10000, "item_id": 50000}
            categorical_embed_dim: 每个离散特征的embedding维度
            continuous_features_dim: Dict[连续特征名, 维度]，例如 {"user_age": 1, "price": 1, "user_hist_prices": 1}
                - 用于自动计算总输入维度和important_features维度
                - 对于序列特征，维度是pooling后的维度
            sequence_features: List[序列特征名]，例如 ["user_hist_items", "user_hist_categories"]
            sequence_pooling: 序列特征的pooling方式，"mean" 或 "max"
            important_features: 重要特征列表，用于生成用户向量u
                - 维度会自动计算，无需手动指定
        
        特征类型说明（4种组合）：
        1. 离散单值特征：在categorical_features中，不在sequence_features中
           - 例如：user_id, item_id
           - 定义：categorical_features={"user_id": 10000}
           - 处理：embedding -> [B, embed_dim]
        
        2. 离散序列特征：在categorical_features中，也在sequence_features中
           - 例如：user_hist_item_ids (用户历史浏览的物品ID序列)
           - 定义：categorical_features={"user_hist_item_ids": 50000}, sequence_features=["user_hist_item_ids"]
           - 处理：embedding -> [B, L, embed_dim] -> pooling -> [B, embed_dim]
        
        3. 连续单值特征：在continuous_features_dim中，不在sequence_features中
           - 例如：user_age, price
           - 定义：continuous_features_dim={"user_age": 1, "price": 1}
           - 处理：直接使用 [B, dim]
        
        4. 连续序列特征：在continuous_features_dim中，也在sequence_features中
           - 例如：user_hist_prices (用户历史浏览的价格序列)
           - 定义：continuous_features_dim={"user_hist_prices": 1}, sequence_features=["user_hist_prices"]
           - 处理：pooling -> [B, dim]
        
        注意：所有特征维度会自动计算，不需要手动指定 dense_input_dim 或 important_features_dim
        """
        super().__init__()

        if expert_hidden_dims is None:
            expert_hidden_dims = [256, 128]
        if tower_hidden_dims is None:
            tower_hidden_dims = [256, 128]

        self.num_domains = num_domains
        self.num_scenarios = num_scenarios
        self.num_tasks = num_tasks
        self.image_dim = image_dim
        self.text_dim = text_dim
        
        # 重要特征列表，用于生成用户向量 u
        self.important_features = important_features if important_features is not None else []
        
        # 离散特征配置
        self.categorical_features = categorical_features if categorical_features is not None else {}
        self.categorical_embed_dim = categorical_embed_dim
        
        # 连续特征配置
        self.continuous_features_dim = continuous_features_dim if continuous_features_dim is not None else {}
        
        # 序列特征配置
        self.sequence_features = sequence_features if sequence_features is not None else []
        self.sequence_pooling = sequence_pooling
        
        # 0. 离散特征的 Embedding 层
        self.categorical_embeddings = nn.ModuleDict()
        for feat_name, vocab_size in self.categorical_features.items():
            self.categorical_embeddings[feat_name] = nn.Embedding(vocab_size, categorical_embed_dim)
        
        # 自动计算总的输入维度
        total_input_dim = self._calculate_total_input_dim()
        
        # 1. Preprocess: 把所有特征直接concat后投影到 embed_dim
        #    处理流程：
        #    - 连续单值特征：直接拼接
        #    - 连续序列特征：pooling后拼接
        #    - 离散单值特征：embedding后拼接
        #    - 离散序列特征：embedding -> pooling后拼接
        self.preprocess = nn.Linear(total_input_dim, embed_dim)
        
        # 1.5 用户向量生成器（从重要特征生成）
        if len(self.important_features) > 0:
            # 自动计算 important_features 的维度
            important_features_dim = self._calculate_important_features_dim()
            self.user_vector_generator = nn.Linear(important_features_dim, embed_dim)
        else:
            # 如果没有指定重要特征，则使用全部特征（向后兼容）
            self.user_vector_generator = None

        # 2. CDI Extractor
        #    为了简单，这里假设 H_intra^d 和 H_share 都由同一个序列特征构造，
        #    实际用的时候你可以在 forward 里替换为真正的序列。
        self.cdi_extractor = CDIExtractor(
            num_domains=num_domains,
            u_dim=embed_dim,
            ctx_dim=embed_dim,
            hidden_dim=embed_dim,
            cdir_dim=cdir_dim,
            qkv_dim=qkv_dim
        )

        # 3. CDI Injector：Norm（多个位置使用）
        # 第一个：在 preprocess 之后
        self.cdir_injector_norm_1 = CDIRInjector_Norm(
            hidden_dim=embed_dim,
            cdir_dim=cdir_dim,
            num_domains=num_domains
        )
        # 第二个：在 cross-net 之后
        self.cdir_injector_norm_2 = CDIRInjector_Norm(
            hidden_dim=embed_dim,
            cdir_dim=cdir_dim,
            num_domains=num_domains
        )
        # 第三个：在加权求和之后
        self.cdir_injector_norm_3 = CDIRInjector_Norm(
            hidden_dim=expert_hidden_dims[-1],  # 注意这里是expert输出的维度
            cdir_dim=cdir_dim,
            num_domains=num_domains
        )
        
        # 3.5 Cross Network (DCNv2)
        self.cross_net = CrossNetV2(
            input_dim=embed_dim,
            num_layers=2,
            low_rank=embed_dim // 4
        )

        # 4. CDI Injector：Gate
        self.domain_gate = CDIRInjector_Gate(
            cdir_dim=cdir_dim,
            num_experts=num_domain_experts
        )
        self.scenario_gate = CDIRInjector_Gate(
            cdir_dim=cdir_dim,
            num_experts=num_scenario_experts
        )
        self.task_gate = CDIRInjector_Gate(
            cdir_dim=cdir_dim,
            num_experts=num_task_experts
        )
        self.image_gate = CDIRInjector_Gate(
            cdir_dim=cdir_dim,
            num_experts=num_image_experts
        )
        self.text_gate = CDIRInjector_Gate(
            cdir_dim=cdir_dim,
            num_experts=num_text_experts
        )

        # 5. Experts
        self.domain_expert = MoEExperts(
            num_experts=num_domain_experts,
            in_dim=embed_dim,
            expert_hidden_dims=expert_hidden_dims
        )
        self.scenario_expert = MoEExperts(
            num_experts=num_scenario_experts,
            in_dim=embed_dim,
            expert_hidden_dims=expert_hidden_dims
        )
        self.task_expert = MoEExperts(
            num_experts=num_task_experts,
            in_dim=embed_dim,
            expert_hidden_dims=expert_hidden_dims
        )
        self.image_expert = MoEExperts(
            num_experts=num_image_experts,
            in_dim=image_dim,
            expert_hidden_dims=expert_hidden_dims
        )
        self.text_expert = MoEExperts(
            num_experts=num_text_experts,
            in_dim=text_dim,
            expert_hidden_dims=expert_hidden_dims
        )

        expert_out_dim = expert_hidden_dims[-1]
        
        # 可学习的加权参数 alpha，用于加权求和5个expert的输出
        self.alpha = nn.Parameter(torch.ones(5) / 5.0)  # 初始化为均匀分布

        # 6. 上层 Tower（对三个任务共用输入，不同 tower）
        # 现在是加权求和，所以输入维度就是 expert_out_dim
        tower_input_dim = expert_out_dim
        tower_dims = [tower_input_dim] + tower_hidden_dims + [1]

        self.tower_click = MLP(tower_dims, last_activation=False)
        self.tower_play = MLP(tower_dims, last_activation=False)
        self.tower_pay = MLP(tower_dims, last_activation=False)

        # 7. BCE Loss
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    # ------------------------
    # 维度计算辅助函数
    # ------------------------
    
    def _calculate_total_input_dim(self) -> int:
        """
        自动计算所有特征concat后的总维度
        """
        total_dim = 0
        
        # 1. 连续单值特征
        for feat_name, feat_dim in self.continuous_features_dim.items():
            if feat_name not in self.sequence_features:
                total_dim += feat_dim
        
        # 2. 连续序列特征 (pooling后)
        for feat_name in self.sequence_features:
            if feat_name not in self.categorical_features and feat_name in self.continuous_features_dim:
                total_dim += self.continuous_features_dim[feat_name]
        
        # 3. 离散单值特征 (embedding后)
        for feat_name in self.categorical_features.keys():
            if feat_name not in self.sequence_features:
                total_dim += self.categorical_embed_dim
        
        # 4. 离散序列特征 (embedding -> pooling后)
        for feat_name in self.sequence_features:
            if feat_name in self.categorical_features:
                total_dim += self.categorical_embed_dim
        
        return total_dim
    
    def _calculate_important_features_dim(self) -> int:
        """
        自动计算重要特征concat后的维度
        """
        total_dim = 0
        
        for feat_name in self.important_features:
            # 判断特征类型
            is_categorical = feat_name in self.categorical_features
            is_sequence = feat_name in self.sequence_features
            
            if is_categorical:
                # 离散特征（单值或序列）都embedding后是 categorical_embed_dim
                total_dim += self.categorical_embed_dim
            elif feat_name in self.continuous_features_dim:
                # 连续特征（单值或序列pooling后）
                total_dim += self.continuous_features_dim[feat_name]
            else:
                raise ValueError(
                    f"Feature '{feat_name}' in important_features is not defined in "
                    f"categorical_features or continuous_features_dim"
                )
        
        return total_dim
    
    # ------------------------
    # feature 处理相关的辅助函数
    # ------------------------
    
    def _pool_sequence_features(self, all_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        对序列特征进行 pooling 并拼接
        处理两种类型：
        1. 连续序列特征：直接对序列进行pooling
        2. 离散序列特征：先embedding，再对embedding序列进行pooling
        返回: [B, total_seq_dim] 或 None
        """
        if len(self.sequence_features) == 0:
            return None
        
        pooled_features = []
        for feat_name in self.sequence_features:
            if feat_name not in all_features:
                raise KeyError(f"Sequence feature '{feat_name}' not found in all_features.")
            
            seq_feat = all_features[feat_name]  # [B, L] 或 [B, L, D]
            
            # 判断是否是离散序列特征
            if feat_name in self.categorical_features:
                # 离散序列特征：[B, L] -> embedding -> [B, L, embed_dim] -> pooling -> [B, embed_dim]
                seq_ids = seq_feat.long()  # [B, L]
                if seq_ids.dim() != 2:
                    raise ValueError(f"Discrete sequence feature '{feat_name}' should be 2D [B, L], got {seq_ids.dim()}D")
                
                # 对序列中的每个ID进行embedding
                seq_embeds = self.categorical_embeddings[feat_name](seq_ids)  # [B, L, embed_dim]
                
                # Pooling over sequence dimension
                if self.sequence_pooling == "mean":
                    pooled = seq_embeds.mean(dim=1)  # [B, embed_dim]
                elif self.sequence_pooling == "max":
                    pooled = seq_embeds.max(dim=1)[0]  # [B, embed_dim]
                else:
                    raise ValueError(f"Unsupported pooling method: {self.sequence_pooling}")
            else:
                # 连续序列特征：[B, L, D] 或 [B, L] -> pooling -> [B, D] 或 [B]
                if seq_feat.dim() == 2:
                    seq_feat = seq_feat.unsqueeze(-1)  # [B, L] -> [B, L, 1]
                
                # Pooling over sequence dimension
                if self.sequence_pooling == "mean":
                    pooled = seq_feat.mean(dim=1)  # [B, D]
                elif self.sequence_pooling == "max":
                    pooled = seq_feat.max(dim=1)[0]  # [B, D]
                else:
                    raise ValueError(f"Unsupported pooling method: {self.sequence_pooling}")
            
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=-1)  # [B, total_seq_dim]
    
    def _embed_categorical_features(self, all_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        对单值离散特征进行 embedding 并拼接
        注意：序列离散特征在 _pool_sequence_features 中单独处理
        返回: [B, num_categorical_single * categorical_embed_dim]
        """
        if len(self.categorical_features) == 0:
            return None
        
        categorical_embeds = []
        for feat_name in self.categorical_features.keys():
            # 跳过序列离散特征（它们在 _pool_sequence_features 中处理）
            if feat_name in self.sequence_features:
                continue
                
            if feat_name not in all_features:
                raise KeyError(f"Categorical feature '{feat_name}' not found in all_features.")
            
            feat_ids = all_features[feat_name].long()  # [B] or [B, 1]
            if feat_ids.dim() == 2:
                feat_ids = feat_ids.squeeze(-1)  # [B]
            elif feat_ids.dim() > 2:
                raise ValueError(f"Categorical feature '{feat_name}' should be 1D or 2D for single value, got {feat_ids.dim()}D. "
                                 f"If it's a sequence feature, add it to sequence_features list.")
            
            embed = self.categorical_embeddings[feat_name](feat_ids)  # [B, categorical_embed_dim]
            categorical_embeds.append(embed)
        
        if len(categorical_embeds) == 0:
            return None
            
        return torch.cat(categorical_embeds, dim=-1)  # [B, num_categorical_single * categorical_embed_dim]

    def _pack_important_features(self, all_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从 all_features 中提取 important_features 列表中指定的特征并拼接
        支持4种类型：
        1. 离散单值特征：embedding
        2. 离散序列特征：embedding -> pooling
        3. 连续单值特征：直接使用
        4. 连续序列特征：pooling
        返回: [B, important_features_dim]
        """
        if len(self.important_features) == 0:
            raise ValueError("important_features list is empty!")
        
        important_list = []
        for key in self.important_features:
            if key not in all_features:
                raise KeyError(f"Important feature '{key}' not found in all_features.")
            
            feat = all_features[key]
            
            # 判断特征类型
            is_categorical = key in self.categorical_features
            is_sequence = key in self.sequence_features
            
            if is_categorical and is_sequence:
                # 类型1：离散序列特征 [B, L] -> embedding -> [B, L, embed_dim] -> pooling -> [B, embed_dim]
                seq_ids = feat.long()  # [B, L]
                seq_embeds = self.categorical_embeddings[key](seq_ids)  # [B, L, embed_dim]
                
                if self.sequence_pooling == "mean":
                    v = seq_embeds.mean(dim=1)  # [B, embed_dim]
                elif self.sequence_pooling == "max":
                    v = seq_embeds.max(dim=1)[0]  # [B, embed_dim]
                else:
                    raise ValueError(f"Unsupported pooling method: {self.sequence_pooling}")
                    
            elif is_categorical and not is_sequence:
                # 类型2：离散单值特征 [B] -> embedding -> [B, embed_dim]
                feat_ids = feat.long()  # [B] or [B, 1]
                if feat_ids.dim() == 2:
                    feat_ids = feat_ids.squeeze(-1)  # [B]
                v = self.categorical_embeddings[key](feat_ids)  # [B, categorical_embed_dim]
                
            elif not is_categorical and is_sequence:
                # 类型3：连续序列特征 [B, L, D] 或 [B, L] -> pooling -> [B, D]
                seq_feat = feat  # [B, L, D] 或 [B, L]
                if seq_feat.dim() == 2:
                    seq_feat = seq_feat.unsqueeze(-1)  # [B, L, 1]
                
                if self.sequence_pooling == "mean":
                    v = seq_feat.mean(dim=1)  # [B, D]
                elif self.sequence_pooling == "max":
                    v = seq_feat.max(dim=1)[0]  # [B, D]
                else:
                    raise ValueError(f"Unsupported pooling method: {self.sequence_pooling}")
                    
            else:
                # 类型4：连续单值特征 [B] 或 [B, D] -> 直接使用
                v = feat
                if v.dim() == 1:
                    v = v.unsqueeze(-1)  # [B, 1]
            
            important_list.append(v)
        
        important_x = torch.cat(important_list, dim=-1)
        return important_x

    def _pack_dense_features(self, all_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        把所有单值连续特征concat成一个向量
        注意：
        - 离散单值特征：通过 _embed_categorical_features 处理
        - 离散序列特征：在 _pool_sequence_features 中embedding + pooling
        - 连续序列特征：在 _pool_sequence_features 中pooling
        - 连续单值特征：这里处理，直接拼接
        """
        dense_list = []
        for k, v in all_features.items():
            # 跳过 label
            if k.startswith("is_click") or k.startswith("is_play") or k.startswith("is_pay"):
                continue
            # 跳过 domain/scenario id（这些单独处理）
            if k in ["domain_id", "scenario_id"]:
                continue
            # 跳过所有离散特征（单值的通过embedding处理，序列的通过pooling处理）
            if k in self.categorical_features:
                continue
            # 跳过所有序列特征（连续序列特征会在 _pool_sequence_features 中处理）
            if k in self.sequence_features:
                continue
            # 跳过 image 和 text features（这些单独处理）
            if k in ["image_features", "text_features"]:
                continue
            
            # 连续单值特征
            # v: [B, dim] 或 [B]
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            # 如果是高维tensor(>=3D)，跳过（应该在sequence_features中定义）
            elif v.dim() >= 3:
                continue
            
            dense_list.append(v)
        
        if len(dense_list) == 0:
            # 如果没有单值连续特征，返回None
            return None
        
        dense_x = torch.cat(dense_list, dim=-1)
        return dense_x

    def _get_labels(self, all_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从字典中抽取 click / play / pay 三个 label。
        这里假设 key 以 'is_click', 'is_play', 'is_pay' 开头。
        """
        def find_label(prefix: str) -> torch.Tensor:
            for k, v in all_features.items():
                if k.startswith(prefix):
                    if v.dim() == 2 and v.size(-1) == 1:
                        return v.squeeze(-1).float()
                    return v.float()
            raise KeyError(f"Label with prefix {prefix} not found in features.")

        y_click = find_label("is_click")
        y_play = find_label("is_play")
        y_pay = find_label("is_pay")
        return y_click, y_play, y_pay


    # ------------------------
    # forward
    # ------------------------

    def forward(self, all_features: Dict[str, torch.Tensor]):
        """
        返回：
            loss: 标量
            click_prob, play_prob, pay_prob: [B] in [0,1]
        """
        device = next(self.parameters()).device

        # 1. 取出 domain / scenario id（整型）
        #    这里假设存在 'domain_id', 'scenario_id' 这两个 key
        #    若你的名字不同，改这里即可
        domain_ids = all_features.get("domain_id", None)
        scenario_ids = all_features.get("scenario_id", None)
        if domain_ids is None or scenario_ids is None:
            raise KeyError("Expected 'domain_id' and 'scenario_id' in all_features.")
        domain_ids = domain_ids.to(device).long()
        scenario_ids = scenario_ids.to(device).long()

        # 2. 特征预处理（4种特征类型的统一处理）
        feature_parts = []
        
        # 2.1 处理连续单值特征（直接拼接）
        dense_x = self._pack_dense_features(all_features)
        if dense_x is not None:
            feature_parts.append(dense_x.to(device))  # [B, Din_continuous_single]
        
        # 2.2 处理所有序列特征（包括连续序列和离散序列）
        #     - 连续序列特征：直接pooling
        #     - 离散序列特征：embedding -> pooling
        sequence_pooled = self._pool_sequence_features(all_features)
        if sequence_pooled is not None:
            feature_parts.append(sequence_pooled.to(device))  # [B, Din_sequence_pooled]
        
        # 2.3 处理离散单值特征（embedding后拼接）
        categorical_embeds = self._embed_categorical_features(all_features)
        if categorical_embeds is not None:
            feature_parts.append(categorical_embeds.to(device))  # [B, num_cat_single * cat_embed_dim]
        
        # 2.4 拼接所有特征
        # 最终拼接顺序：连续单值 + 序列pooling(连续+离散) + 离散单值embedding
        if len(feature_parts) == 0:
            raise ValueError("No valid features found in all_features!")
        
        all_features_concat = torch.cat(feature_parts, dim=-1)  # [B, total_dim]

        # 2.5 统一投影到 embed_dim
        h0 = self.preprocess(all_features_concat)  # [B, De]
        h0 = F.relu(h0)
        
        # 2.5 生成用户向量 u（仅使用重要特征）
        if self.user_vector_generator is not None and len(self.important_features) > 0:
            # 使用指定的重要特征生成用户向量
            important_x = self._pack_important_features(all_features).to(device)  # [B, important_dim]
            u = self.user_vector_generator(important_x)  # [B, De]
            u = F.relu(u)
        else:
            # 向后兼容：如果没有指定重要特征，使用全部特征表征
            u = h0

        # 3. 构造 CDI Extractor 的输入
        #    这里只是一个示意：用相同的序列当作 H_intra^d 和 H_share。
        #    实际中你应该用真正的序列行为特征。
        B = h0.size(0)
        # toy 序列：复制若干次
        L_seq = 4
        base_seq = h0.unsqueeze(1).repeat(1, L_seq, 1)  # [B,L,De]

        H_intra = {d: base_seq for d in range(self.num_domains)}
        H_share = base_seq

        cdir_dict, L_con = self.cdi_extractor(
            u=u,
            H_intra=H_intra,
            H_share=H_share,
            ctx=None
        )

        # 4. 第一个 CDI Injector Norm
        h1 = self.cdir_injector_norm_1(
            h=h0,
            cdir_dict=cdir_dict,
            domain_ids=domain_ids
        )  # [B,De]
        
        # 5. Cross Network (DCNv2)
        h2 = self.cross_net(h1)  # [B,De]
        
        # 6. 第二个 CDI Injector Norm
        h3 = self.cdir_injector_norm_2(
            h=h2,
            cdir_dict=cdir_dict,
            domain_ids=domain_ids
        )  # [B,De]

        # 7. CDI Gate for domain / scenario / task
        #    对于 domain/scenario，我们选用其对应的 CDIR
        z_domain = torch.zeros(B, next(iter(cdir_dict.values())).size(-1), device=device)
        for d in range(self.num_domains):
            idx = (domain_ids == d)
            if not idx.any():
                continue
            z_domain[idx] = cdir_dict[d][idx]
        gate_domain = self.domain_gate(z_domain)  # [B, N_domain_expert]

        # scenario 这里为了简单直接复用 z_domain，也可以额外构造 scenario 对应的 CDIR
        z_scenario = z_domain
        gate_scenario = self.scenario_gate(z_scenario)  # [B,N_scenario_expert]

        # task gate：对三个任务共用一套 gate（或者你可以为每个 task 单独建一个 gate）
        # 这里简单用用户级的 pool(z_domain) 作为输入
        gate_task = self.task_gate(z_domain)  # [B,N_task_expert]

        # 8. 处理 image 和 text features
        image_features = all_features.get("image_features", None)
        text_features = all_features.get("text_features", None)
        
        if image_features is None or text_features is None:
            raise KeyError("Expected 'image_features' and 'text_features' in all_features.")
        
        image_features = image_features.to(device).float()  # [B, 64]
        text_features = text_features.to(device).float()    # [B, 64]
        
        # 为 image 和 text 生成 gate
        gate_image = self.image_gate(z_domain)  # [B, N_image_expert]
        gate_text = self.text_gate(z_domain)    # [B, N_text_expert]
        
        # 9. MoE Experts
        # domain, scenario, task expert 使用经过 Cross-Net 处理后的 h3
        h_domain = self.domain_expert(h3, gate_domain)          # [B, Dexp]
        h_scenario = self.scenario_expert(h3, gate_scenario)    # [B, Dexp]
        h_task = self.task_expert(h3, gate_task)                # [B, Dexp]
        # image 和 text expert 使用原始的特征
        h_image = self.image_expert(image_features, gate_image) # [B, Dexp]
        h_text = self.text_expert(text_features, gate_text)     # [B, Dexp]

        # 10. 使用可学习的 alpha 进行加权求和
        # 先对 alpha 做 softmax 归一化
        alpha_normalized = F.softmax(self.alpha, dim=0)  # [5]
        
        # 堆叠所有 expert 输出
        expert_outputs = torch.stack([h_domain, h_scenario, h_task, h_image, h_text], dim=1)  # [B, 5, Dexp]
        
        # 加权求和
        h_weighted = (expert_outputs * alpha_normalized.view(1, 5, 1)).sum(dim=1)  # [B, Dexp]
        
        # 11. 第三个 CDI Injector Norm（在加权求和之后）
        h_final = self.cdir_injector_norm_3(
            h=h_weighted,
            cdir_dict=cdir_dict,
            domain_ids=domain_ids
        )  # [B, Dexp]

        # 12. 三个任务的 tower 输出 logits
        logit_click = self.tower_click(h_final).squeeze(-1)  # [B]
        logit_play = self.tower_play(h_final).squeeze(-1)
        logit_pay = self.tower_pay(h_final).squeeze(-1)

        # 13. 计算 BCE loss
        y_click, y_play, y_pay = self._get_labels(all_features)
        y_click = y_click.to(device)
        y_play = y_play.to(device)
        y_pay = y_pay.to(device)

        loss_click = self.bce(logit_click, y_click)
        loss_play = self.bce(logit_play, y_play)
        loss_pay = self.bce(logit_pay, y_pay)

        # 加上 CDI 的对比损失
        loss = loss_click + loss_play + loss_pay + L_con

        # 14. 输出 0-1 之间的概率
        click_prob = torch.sigmoid(logit_click)
        play_prob = torch.sigmoid(logit_play)
        pay_prob = torch.sigmoid(logit_pay)

        return {
            "loss": loss,
            "click_prob": click_prob,
            "play_prob": play_prob,
            "pay_prob": pay_prob
        }
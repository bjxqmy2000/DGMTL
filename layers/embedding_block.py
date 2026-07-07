
from torch import nn
import torch

class Embedding_Block(nn.Module):
    """
    差异化嵌入模块：为每个任务生成特定嵌入 + 共享嵌入（可选）
    
    设计理念：
    - 任务特定嵌入（可选）：捕捉不同QoS属性的差异化特征敏感度
      例如：RT对物理位置更敏感，TP对带宽更敏感
    - 共享嵌入：提取所有任务共享的通用特征
      例如：网络拓扑、服务提供商等
    
    输入:
        user_features: 用户特征张量 (batch_size, num_user_features)
        ws_features: Web服务特征张量 (batch_size, num_ws_features)
    
    输出:
        List[Tensor]，长度为 num_tasks + 1:
        - [0 ~ num_tasks-1]: 各任务的特定嵌入（如果use_task_specific=False，则复用共享嵌入）
        - [num_tasks]: 共享嵌入
        每个嵌入的shape: (batch_size, embedding_dim * 总特征数)
    """
    def __init__(self, 
                 user_feature_info,
                 ws_feature_info, 
                 embedding_dim,
                 num_tasks,
                 use_task_specific=True):
        """
        初始化差异化嵌入模块
        
        参数:
            user_feature_info: 用户特征信息字典 {特征名: 类别数}
            ws_feature_info: Web服务特征信息字典 {特征名: 类别数}
            embedding_dim: 每个特征的嵌入维度
            num_tasks: 任务数量
            use_task_specific: 是否使用任务特定嵌入（默认True）
        """
        super().__init__()
        self.user_keys = list(user_feature_info.keys())
        self.ws_keys = list(ws_feature_info.keys())
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        self.use_task_specific = use_task_specific

        # ========== 1. 共享嵌入层 ==========
        # 所有任务共享的嵌入表，用于提取通用特征
        self.user_shared_embed = nn.ModuleDict({
            feat: nn.Embedding(num, embedding_dim)
            for feat, num in user_feature_info.items()
        })
        self.ws_shared_embed = nn.ModuleDict({
            feat: nn.Embedding(num, embedding_dim)
            for feat, num in ws_feature_info.items()
        })

        # ========== 2. 任务特定嵌入层（可选） ==========
        # 每个任务有自己的嵌入表，用于捕捉任务独特的特征模式
        if self.use_task_specific:
            self.user_task_embeds = nn.ModuleList([
                nn.ModuleDict({
                    feat: nn.Embedding(num, embedding_dim)
                    for feat, num in user_feature_info.items()
                }) for _ in range(num_tasks)
            ])
            self.ws_task_embeds = nn.ModuleList([
                nn.ModuleDict({
                    feat: nn.Embedding(num, embedding_dim)
                    for feat, num in ws_feature_info.items()
                }) for _ in range(num_tasks)
            ])
        else:
            self.user_task_embeds = None
            self.ws_task_embeds = None

    def forward(self, user_features, ws_features):
        """
        前向传播：生成差异化嵌入
        
        参数:
            user_features: (batch_size, num_user_features)
            ws_features: (batch_size, num_ws_features)
        
        返回:
            [任务1嵌入, 任务2嵌入, ..., 共享嵌入]
        """
        # ========== 步骤1: 生成共享嵌入 ==========
        # 对每个用户特征，使用共享嵌入表进行编码
        user_shared = [
            self.user_shared_embed[k](user_features[:, idx].long()) 
            for idx, k in enumerate(self.user_keys)
        ]
        # 对每个Web服务特征，使用共享嵌入表进行编码
        ws_shared = [
            self.ws_shared_embed[k](ws_features[:, idx].long()) 
            for idx, k in enumerate(self.ws_keys)
        ]
        
        # ========== 张量操作: torch.cat() ==========
        # 拼接所有特征的嵌入
        # 输入: user_shared有5个张量，每个(batch, 128)
        #      ws_shared有8个张量，每个(batch, 128)
        # 输出: (batch, 128*13) = (batch, 1664)
        # 
        # 例如: batch=4, embedding_dim=128
        # user_shared = [(4,128), (4,128), (4,128), (4,128), (4,128)]
        # ws_shared   = [(4,128), (4,128), ..., (4,128)]  # 8个
        # 
        # cat操作沿着dim=1拼接:
        # (4,128) + (4,128) + ... + (4,128) → (4, 128*13) = (4, 1664)
        #                                           ↑
        #                                    13个特征的嵌入拼接
        shared_embeddings = torch.cat(user_shared + ws_shared, dim=1)

        # ========== 步骤2: 生成任务特定嵌入 ==========
        task_embeddings = []
        if self.use_task_specific:
            # 使用任务特定嵌入
            for i in range(self.num_tasks):
                # 任务i使用自己的嵌入表
                user_task = [
                    self.user_task_embeds[i][k](user_features[:, idx].long()) 
                    for idx, k in enumerate(self.user_keys)
                ]
                ws_task = [
                    self.ws_task_embeds[i][k](ws_features[:, idx].long()) 
                    for idx, k in enumerate(self.ws_keys)
                ]
                # 拼接任务i的所有特征嵌入（与共享嵌入相同的操作）
                task_emb = torch.cat(user_task + ws_task, dim=1)
                task_embeddings.append(task_emb)
        else:
            # 不使用任务特定嵌入，所有任务复用共享嵌入
            for i in range(self.num_tasks):
                task_embeddings.append(shared_embeddings)

        # ========== 返回：任务特定嵌入 + 共享嵌入 ==========
        return task_embeddings + [shared_embeddings]

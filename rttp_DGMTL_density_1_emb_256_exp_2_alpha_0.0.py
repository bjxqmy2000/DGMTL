import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import logging
import time
import math
import pandas as pd
import copy
from layers.mlp_block import MLP_Block
from layers.embedding_block import Embedding_Block
from layers.gradnorm_weight_block import GradNormWeighting

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Dataset:
    """数据集类，处理RT和TP数据"""

    def __init__(self, density, userlist_path, wslist_path, data_dir):
        # 记录用户有效特征的编码数量
        self.user_id_num = 0
        self.user_country_num = 0
        self.user_as_num = 0
        self.user_latitude_num = 0
        self.user_longitude_num = 0

        # 记录服务有效特征的编码数量
        self.ws_id_num = 0
        self.ws_wsdl_address_num = 0
        self.ws_provider_num = 0
        self.ws_ip_address_num = 0
        self.ws_country_num = 0
        self.ws_as_num = 0
        self.ws_latitude_num = 0
        self.ws_longitude_num = 0

        # 文件路径
        self.userlist_path = userlist_path
        self.wslist_path = wslist_path
        self.data_dir = data_dir

        # 设置RT和TP矩阵路径（训练集、验证集、测试集）
        rtMatrix_train_path = os.path.join(data_dir, f"sparse/rtMatrix{density}_train.csv")
        rtMatrix_val_path = os.path.join(data_dir, f"sparse/rtMatrix{density}_val.csv")
        rtMatrix_test_path = os.path.join(data_dir, f"sparse/rtMatrix{density}_test.csv")
        tpMatrix_train_path = os.path.join(data_dir, f"sparse/tpMatrix{density}_train.csv")
        tpMatrix_val_path = os.path.join(data_dir, f"sparse/tpMatrix{density}_val.csv")
        tpMatrix_test_path = os.path.join(data_dir, f"sparse/tpMatrix{density}_test.csv")

        # 读取用户和服务特征
        userlist_df = pd.read_csv(userlist_path)
        wslist_df = pd.read_csv(wslist_path)

        # 读取训练、验证和测试矩阵
        rtMatrix_train_df = pd.read_csv(rtMatrix_train_path, header=None)
        rtMatrix_val_df = pd.read_csv(rtMatrix_val_path, header=None)
        rtMatrix_test_df = pd.read_csv(rtMatrix_test_path, header=None)
        tpMatrix_train_df = pd.read_csv(tpMatrix_train_path, header=None)
        tpMatrix_val_df = pd.read_csv(tpMatrix_val_path, header=None)
        tpMatrix_test_df = pd.read_csv(tpMatrix_test_path, header=None)

        logger.info("文件预读取完毕")

        # 获取用户和服务信息
        self.user_info = self.get_user_info(userlist_df)
        self.ws_info = self.get_ws_info(wslist_df)

        # 获取矩阵数据
        self.train_rt_info = self.get_rt_info(rtMatrix_train_df)
        self.val_rt_info = self.get_rt_info(rtMatrix_val_df)
        self.test_rt_info = self.get_rt_info(rtMatrix_test_df)
        self.train_tp_info = self.get_tp_info(tpMatrix_train_df)
        self.val_tp_info = self.get_tp_info(tpMatrix_val_df)
        self.test_tp_info = self.get_tp_info(tpMatrix_test_df)

        # 构建数据集
        self.build_dataset()

    def get_user_info(self, userlist_df):
        """获取并编码用户特征"""
        userlist_df["user_id"] = userlist_df["user_id"].astype(int)

        # user_id，ip_address，ip_number具有唯一性，故只使用user_id
        other_features = ["country", "as", "latitude", "longitude"]
        for feat in other_features:
            lbe = LabelEncoder()
            userlist_df[feat] = lbe.fit_transform(userlist_df[feat])

        max_values = userlist_df[["user_id", "country", "as", "latitude", "longitude"]].max()
        self.user_id_num = max_values["user_id"] + 1
        self.user_country_num = max_values["country"] + 1
        self.user_as_num = max_values["as"] + 1
        self.user_latitude_num = max_values["latitude"] + 1
        self.user_longitude_num = max_values["longitude"] + 1

        user_info = {
            row["user_id"]: {
                'user_id': row["user_id"],
                'country': row["country"],
                'as': row["as"],
                'latitude': row["latitude"],
                'longitude': row["longitude"]
            }
            for _, row in userlist_df.iterrows()
        }

        logger.info("user_info获取完毕")
        return user_info

    def get_ws_info(self, wslist_df):
        """获取并编码Web服务特征"""
        wslist_df["ws_id"] = wslist_df["ws_id"].astype(int)

        # ip_address，ip_number具有唯一性，故只使用ip_address
        other_features = ["wsdl_address", "provider", "ip_address",
                          "country", "as", "latitude", "longitude"]
        for feat in other_features:
            lbe = LabelEncoder()
            wslist_df[feat] = lbe.fit_transform(wslist_df[feat])

        max_values = wslist_df[
            ["ws_id", "wsdl_address", "provider", "ip_address",
             "country", "as", "latitude", "longitude"]].max()
        self.ws_id_num = max_values["ws_id"] + 1
        self.ws_wsdl_address_num = max_values["wsdl_address"] + 1
        self.ws_provider_num = max_values["provider"] + 1
        self.ws_ip_address_num = max_values["ip_address"] + 1
        self.ws_country_num = max_values["country"] + 1
        self.ws_as_num = max_values["as"] + 1
        self.ws_latitude_num = max_values["latitude"] + 1
        self.ws_longitude_num = max_values["longitude"] + 1

        ws_info = {
            row["ws_id"]: {
                'ws_id': row["ws_id"],
                'wsdl_address': row["wsdl_address"],
                'provider': row["provider"],
                'ip_address': row["ip_address"],
                'country': row["country"],
                'as': row["as"],
                'latitude': row["latitude"],
                'longitude': row["longitude"]
            }
            for _, row in wslist_df.iterrows()
        }

        logger.info("ws_info获取完毕")
        return ws_info

    def get_rt_info(self, rtMatrix_df):
        """处理响应时间矩阵"""
        rtMatrix_array = rtMatrix_df.values
        rows, cols = np.where(rtMatrix_array != -1)
        rts = rtMatrix_array[rows, cols]

        rt_info = {}
        for i, j, rt in zip(rows, cols, rts):
            if i not in rt_info:
                rt_info[i] = {}
            rt_info[i][j] = float(rt)

        logger.info("rt_info获取完毕，非零元素数量：{}".format(len(rts)))
        return rt_info

    def get_tp_info(self, tpMatrix_df):
        """处理吞吐量矩阵"""
        tpMatrix_array = tpMatrix_df.values
        rows, cols = np.where(tpMatrix_array != -1)
        tps = tpMatrix_array[rows, cols]

        tp_info = {}
        for i, j, tp in zip(rows, cols, tps):
            if i not in tp_info:
                tp_info[i] = {}
            tp_info[i][j] = float(tp)

        logger.info("tp_info获取完毕，非零元素数量：{}".format(len(tps)))
        return tp_info

    def build_dataset(self):
        """构建数据集"""
        # 构建训练集
        self.train_dataset = self.get_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.train_rt_info,
            tp_info=self.train_tp_info
        )

        # 构建验证集
        self.val_dataset = self.get_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.val_rt_info,
            tp_info=self.val_tp_info
        )

        # 构建测试集
        self.test_dataset = self.get_dataset(
            user_info=self.user_info,
            ws_info=self.ws_info,
            rt_info=self.test_rt_info,
            tp_info=self.test_tp_info
        )

        logger.info(f"dataset statistics:")
        logger.info(f"train dataset size: {len(self.train_dataset)}")
        logger.info(f"val dataset size: {len(self.val_dataset)}")
        logger.info(f"test dataset size: {len(self.test_dataset)}")

    def get_dataset(self, user_info, ws_info, rt_info, tp_info):
        """构建数据集：允许只含 RT 或 TP，但至少包含一个 QoS"""

        # 收集所有"有效交互"(user_id, ws_id)：rt!=0 或 tp!=0
        pairs = set()
        for user_id, services in rt_info.items():
            for ws_id in services.keys():
                pairs.add((user_id, ws_id))
        for user_id, services in tp_info.items():
            for ws_id in services.keys():
                pairs.add((user_id, ws_id))

        pairs = list(pairs)
        logger.info(f"有效样本对数量(至少包含RT或TP): {len(pairs)}")

        # 构建数据集
        dataset = []
        for user_id, ws_id in pairs:
            has_rt = user_id in rt_info and ws_id in rt_info[user_id]
            has_tp = user_id in tp_info and ws_id in tp_info[user_id]

            # 防御：不应出现两个都没有的无效样本
            if not has_rt and not has_tp:
                continue

            rt_value = rt_info[user_id][ws_id] if has_rt else -1.0
            tp_value = tp_info[user_id][ws_id] if has_tp else -1.0

            item = {
                'user_info': user_info[user_id],
                'ws_info': ws_info[ws_id],
                'rt': rt_value,
                'tp': tp_value,
                'rt_present': has_rt,
                'tp_present': has_tp,
            }

            dataset.append(item)

        return dataset


class TorchDataset(torch.utils.data.Dataset):
    """PyTorch数据集类"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 将用户/服务信息转换为 float32 张量（后续在 embedding 中会 .long() 取索引）
        user_info = torch.tensor([
            float(item['user_info']['user_id']),
            float(item['user_info']['country']),
            float(item['user_info']['as']),
            float(item['user_info']['latitude']),
            float(item['user_info']['longitude']),
        ], dtype=torch.float32)

        ws_info = torch.tensor([
            float(item['ws_info']['ws_id']),
            float(item['ws_info']['wsdl_address']),
            float(item['ws_info']['provider']),
            float(item['ws_info']['ip_address']),
            float(item['ws_info']['country']),
            float(item['ws_info']['as']),
            float(item['ws_info']['latitude']),
            float(item['ws_info']['longitude']),
        ], dtype=torch.float32)

        return {
            'user_info': user_info,
            'ws_info': ws_info,
            'rt': torch.tensor(float(item['rt']), dtype=torch.float32),
            'tp': torch.tensor(float(item['tp']), dtype=torch.float32),
            'rt_mask': torch.tensor(1.0 if item.get('rt_present', True) else 0.0, dtype=torch.float32),
            'tp_mask': torch.tensor(1.0 if item.get('tp_present', True) else 0.0, dtype=torch.float32),
        }


class STEM_Layer(nn.Module):
    """
    STEM层实现：包含差异化专家网络和门控机制

    核心设计：
    1. 任务特定专家：每个任务有自己的专家网络，用于捕捉任务独特的特征模式
    2. 共享专家：所有任务共享的专家网络，用于提取通用特征
    3. 门控机制：为每个任务分配一个门控网络，动态选择专家输出的权重
    """

    def __init__(self, num_tasks, num_shared_experts, num_specific_experts,
                 input_dim, expert_hidden_units, gate_hidden_units,
                 batch_norm, hidden_activations, dropout_rates, use_stop_gradient=True):
        super(STEM_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.use_stop_gradient = use_stop_gradient

        # ========== 1. 共享专家网络 ==========
        self.shared_experts = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=expert_hidden_units,
                hidden_activations=hidden_activations,
                batch_norm=batch_norm,
                dropout_rates=dropout_rates
            ) for _ in range(self.num_shared_experts)
        ])

        # ========== 2. 任务特定专家网络 ==========
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                MLP_Block(
                    input_dim=input_dim,
                    hidden_units=expert_hidden_units,
                    hidden_activations=hidden_activations,
                    batch_norm=batch_norm,
                    dropout_rates=dropout_rates
                ) for _ in range(self.num_specific_experts)
            ]) for _ in range(num_tasks)
        ])

        # ========== 3. 门控网络 ==========
        self.gate = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=gate_hidden_units,
                hidden_activations=hidden_activations,
                batch_norm=batch_norm,
                dropout_rates=dropout_rates,
                output_dim=num_specific_experts * num_tasks + num_shared_experts,
                output_activation="softmax"
            ) for i in range(self.num_tasks + 1)
        ])

    def forward(self, x, return_gate=False):
        """STEM层的前向传播：实现"全前向，任务特定后向"的门控机制"""
        specific_expert_outputs = []
        shared_expert_outputs = []

        # 计算所有任务特定专家的输出
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)

        # 计算所有共享专家的输出
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))

        # 门控机制融合专家输出
        stem_outputs = []
        stem_gates = []

        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                # 任务特定门控
                gate_input = []

                for j in range(self.num_tasks):
                    if j == i:
                        gate_input.extend(specific_expert_outputs[j])
                    else:
                        specific_expert_outputs_j = specific_expert_outputs[j]
                        # 根据use_stop_gradient决定是否对其他任务的专家执行停止梯度
                        if self.use_stop_gradient:
                            specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                        gate_input.extend(specific_expert_outputs_j)

                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input, dim=1)
                gate = self.gate[i](x[i])

                if return_gate:
                    specific_gate = gate[:, :self.num_specific_experts * self.num_tasks].mean(0)
                    task_gate = torch.chunk(specific_gate, chunks=self.num_tasks)
                    specific_gate_list = []
                    for tg in task_gate:
                        specific_gate_list.append(torch.sum(tg))
                    shared_gate = gate[:, -self.num_shared_experts:].mean(0).sum()
                    target_task_gate = torch.stack(specific_gate_list + [shared_gate], dim=0).view(-1)
                    assert len(target_task_gate) == self.num_tasks + 1
                    stem_gates.append(target_task_gate)

                gate_expanded = gate.unsqueeze(-1)
                weighted = gate_expanded * gate_input
                stem_output = torch.sum(weighted, dim=1)
                stem_outputs.append(stem_output)
            else:
                # 共享门控
                gate_input = []

                for j in range(self.num_tasks):
                    specific_expert_outputs_j = specific_expert_outputs[j]
                    # 根据use_stop_gradient决定是否对任务特定专家执行停止梯度
                    if self.use_stop_gradient:
                        specific_expert_outputs_j = [out.detach() for out in specific_expert_outputs_j]
                    gate_input.extend(specific_expert_outputs_j)

                gate_input.extend(shared_expert_outputs)
                gate_input = torch.stack(gate_input, dim=1)
                gate = self.gate[i](x[-1])
                stem_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                stem_outputs.append(stem_output)

        if return_gate:
            return stem_outputs, stem_gates
        else:
            return stem_outputs


class STEM(nn.Module):
    """
    STEM模型完整实现：Shared and Task-specific EMbeddings with Multi-task learning
    """

    def __init__(self, user_feature_info, ws_feature_info,
                 num_tasks=2,
                 embedding_dim=16,
                 num_layers=1,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 expert_hidden_units=[256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 batch_norm=True,
                 hidden_activations="relu",
                 dropout_rates=0.0,
                 use_task_specific_embedding=True,
                 use_stop_gradient=True,
                 use_gradnorm=True):
        super(STEM, self).__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.use_task_specific_embedding = use_task_specific_embedding
        self.use_stop_gradient = use_stop_gradient
        self.use_gradnorm = use_gradnorm

        # ========== 1. 差异化嵌入层 ==========
        self.embedding_layer = Embedding_Block(
            user_feature_info,
            ws_feature_info,
            embedding_dim,
            num_tasks,
            use_task_specific=use_task_specific_embedding
        )

        # 计算嵌入层输出的总维度
        num_user_features = len(user_feature_info)
        num_ws_features = len(ws_feature_info)
        self.input_dim = (num_user_features + num_ws_features) * embedding_dim

        # ========== 2. STEM层（可堆叠） ==========
        self.stem_layers = nn.ModuleList([
            STEM_Layer(
                num_shared_experts=num_shared_experts,
                num_specific_experts=num_specific_experts,
                num_tasks=num_tasks,
                input_dim=self.input_dim if i == 0 else expert_hidden_units[-1],
                expert_hidden_units=expert_hidden_units,
                gate_hidden_units=gate_hidden_units,
                batch_norm=batch_norm,
                hidden_activations=hidden_activations,
                dropout_rates=dropout_rates,
                use_stop_gradient=use_stop_gradient
            ) for i in range(num_layers)
        ])

        # ========== 3. 任务塔（预测头） ==========
        self.towers = nn.ModuleList([
            MLP_Block(
                input_dim=expert_hidden_units[-1],
                hidden_units=tower_hidden_units,
                hidden_activations=hidden_activations,
                output_dim=1,
                batch_norm=batch_norm,
                dropout_rates=dropout_rates
            ) for _ in range(self.num_tasks)
        ])

    def get_shared_params(self):
        """获取共享参数（共享嵌入 + 共享专家）"""
        shared_params = []

        # 共享嵌入参数
        if hasattr(self.embedding_layer, 'user_shared_embed'):
            shared_params.extend(self.embedding_layer.user_shared_embed.parameters())
        if hasattr(self.embedding_layer, 'ws_shared_embed'):
            shared_params.extend(self.embedding_layer.ws_shared_embed.parameters())

        # 共享专家参数
        for layer in self.stem_layers:
            for expert in layer.shared_experts:
                shared_params.extend(expert.parameters())

        return shared_params

    def get_last_shared_layer_params(self):
        """
        获取最后一层共享专家的参数（用于GradNorm计算）

        为了节省计算开销，GradNorm通常只选取最后一层共享层的参数来计算梯度范数，
        因为这一层直接决定了共享特征的表达。
        """
        shared_params = []

        # 获取最后一个STEM层的共享专家参数
        last_stem_layer = self.stem_layers[-1]
        for expert in last_stem_layer.shared_experts:
            shared_params.extend(expert.parameters())

        return shared_params

    def forward(self, user_features, ws_features):
        """STEM模型的前向传播"""
        # 差异化嵌入
        stem_inputs = self.embedding_layer(user_features, ws_features)

        # 通过STEM层
        for i in range(self.num_layers):
            stem_outputs = self.stem_layers[i](stem_inputs)
            stem_inputs = stem_outputs

        # 任务塔预测
        preds = []
        for i in range(self.num_tasks):
            preds.append(self.towers[i](stem_outputs[i]))

        return preds


def train_stem_model(model, train_loader, valid_loader, optimizer, scheduler, device,
                     num_epochs, eval_epochs=1, early_stopping_patience=10,
                     gradnorm_alpha=1.5, gradnorm_lr=0.025):
    """
    训练STEM模型

    参数:
        model: STEM模型实例
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备（CPU或GPU）
        num_epochs: 训练轮数
        eval_epochs: 每隔多少轮评估一次
        early_stopping_patience: 早停耐心值（验证集多少轮无提升则停止）
        gradnorm_alpha: GradNorm的alpha超参数（默认1.5）
        gradnorm_lr: GradNorm权重优化器的学习率（默认0.025）

    返回:
        best_metrics: 最佳验证指标字典
    """
    logger.info("开始训练STEM模型")

    criterion = nn.SmoothL1Loss(reduction='none')

    # 根据use_gradnorm决定使用哪种损失加权方法
    if model.use_gradnorm:
        loss_weighting = GradNormWeighting(num_tasks=model.num_tasks, alpha=gradnorm_alpha).to(device)
        # 为权重创建一个独立的优化器！严禁加入主网络optimizer
        weight_optimizer = torch.optim.Adam([loss_weighting.task_weights], lr=gradnorm_lr)
        logger.info(f"使用GradNorm损失加权方法 (alpha={gradnorm_alpha}, lr={gradnorm_lr})")
    else:
        loss_weighting = None
        weight_optimizer = None
        logger.info("不使用损失加权")

    best_metrics = {
        "best_epoch": 0,
        "val_total_loss": float('inf'),
        "rt_mae": float('inf'),
        "rt_rmse": float('inf'),
        "tp_mae": float('inf'),
        "tp_rmse": float('inf'),
    }
    no_improvement_epochs = 0
    best_epoch = 0
    best_state_dict = None

    total_train_time = 0
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        if loss_weighting is not None:
            loss_weighting.train()

        train_loss = 0.0
        rt_losses = 0.0
        tp_losses = 0.0
        batch_count = 0

        for batch in train_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)
            tp_targets = batch['tp'].to(device)
            rt_mask = batch.get('rt_mask', torch.ones_like(rt_targets)).to(device)
            tp_mask = batch.get('tp_mask', torch.ones_like(tp_targets)).to(device)

            # 清理两个优化器的梯度
            optimizer.zero_grad()
            if model.use_gradnorm:
                weight_optimizer.zero_grad()

            # 获取预测结果
            preds = model(user_features, ws_features)
            rt_pred = preds[0].squeeze(-1)
            tp_pred = preds[1].squeeze(-1)

            # 计算单独的损失
            rt_per = criterion(rt_pred, rt_targets)
            tp_per = criterion(tp_pred, tp_targets)

            rt_denom = torch.clamp(rt_mask.sum(), min=1.0)
            tp_denom = torch.clamp(tp_mask.sum(), min=1.0)
            rt_loss = (rt_per * rt_mask).sum() / rt_denom
            tp_loss = (tp_per * tp_mask).sum() / tp_denom

            # 根据use_gradnorm决定如何计算总损失
            if model.use_gradnorm:
                # ==========================================
                # 阶段一：计算 GradNorm 并更新权重
                # ==========================================

                # 1. 获取最后共享层的参数
                last_shared_params = list(model.get_last_shared_layer_params())

                # 2. 前向计算带梯度的 weighted_losses，专门用于给 GradNorm 算高阶导数
                weighted_losses = loss_weighting([rt_loss, tp_loss])

                # 3. 使用Autograd计算GradNorm的损失 (此时不影响网络主干更新)
                loss_gradnorm = loss_weighting.compute_gradnorm_loss(
                    weighted_losses, [rt_loss, tp_loss], last_shared_params
                )

                # 4. 单独为权重参数计算梯度
                weight_optimizer.zero_grad()
                loss_gradnorm.backward(retain_graph=True)
                weight_optimizer.step()

                # 5. 权重更新后执行归一化
                loss_weighting.renormalize_weights()

                # ==========================================
                # 阶段二：使用【最新权重】更新模型主干参数
                # ==========================================

                # 6. 【关键】使用 detached (脱离计算图) 的权重，重新组合出用于主网络更新的总损失。
                detached_weights = loss_weighting.task_weights.detach()
                pure_total_loss = detached_weights[0] * rt_loss + detached_weights[1] * tp_loss

                # 7. 清理主优化器梯度，正常的主网络反向传播与参数更新
                optimizer.zero_grad()
                pure_total_loss.backward()
                optimizer.step()

                # 记录 total_loss 的值用于日志
                total_loss = pure_total_loss

            else:
                # 不使用 GradNorm 的逻辑保持不变
                total_loss = rt_loss + tp_loss
                total_loss.backward()
                optimizer.step()

            train_loss += total_loss.item()
            rt_losses += rt_loss.item()
            tp_losses += tp_loss.item()
            batch_count += 1

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        # 计算平均损失
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_rt_loss = rt_losses / batch_count if batch_count > 0 else 0
        avg_tp_loss = tp_losses / batch_count if batch_count > 0 else 0

        # 计算本轮训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        total_train_time += epoch_time

        # 记录权重信息
        if model.use_gradnorm:
            weights = loss_weighting.get_weights()
            loss_ratios = loss_weighting.get_loss_ratios(
                [torch.tensor(avg_rt_loss, device=device), torch.tensor(avg_tp_loss, device=device)])
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"训练损失: {avg_train_loss:.4f}, "
                f"RT损失: {avg_rt_loss:.4f}, TP损失: {avg_tp_loss:.4f}, "
                f"GradNorm权重: RT={weights[0].item():.4f}, TP={weights[1].item():.4f}, "
                f"损失比率: RT={loss_ratios[0].item():.4f}, TP={loss_ratios[1].item():.4f}, "
                f"训练时间={epoch_time:.2f}秒"
            )
        else:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"训练损失: {avg_train_loss:.4f}, "
                f"RT损失: {avg_rt_loss:.4f}, TP损失: {avg_tp_loss:.4f}, "
                f"训练时间={epoch_time:.2f}秒"
            )

        # 验证
        if (epoch + 1) % eval_epochs == 0:
            valid_start_time = time.time()
            valid_metrics = evaluate_stem_model(
                model, valid_loader, device
            )
            valid_time = time.time() - valid_start_time

            logger.info(
                f"验证损失: {valid_metrics['total_loss']:.4f}, "
                f"RT MAE: {valid_metrics['rt_mae']:.4f}, RT RMSE: {valid_metrics['rt_rmse']:.4f}, "
                f"TP MAE: {valid_metrics['tp_mae']:.4f}, TP RMSE: {valid_metrics['tp_rmse']:.4f}, "
                f"验证时间: {valid_time:.2f}秒"
            )

            if valid_metrics['total_loss'] < best_metrics['val_total_loss']:
                best_metrics = {
                    **valid_metrics,
                    "best_epoch": epoch + 1,
                    "val_total_loss": valid_metrics['total_loss'],
                }
                best_epoch = epoch + 1
                no_improvement_epochs = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                logger.info(f"✓ 找到更好的模型! (Epoch {best_epoch})")
            else:
                no_improvement_epochs += 1
                logger.info(f"× 连续 {no_improvement_epochs} 轮验证未改进")

            if early_stopping_patience > 0 and no_improvement_epochs >= early_stopping_patience:
                logger.info(f"早停触发：连续 {no_improvement_epochs} 轮验证集无提升")
                break

    # 计算平均每轮训练时间
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0

    logger.info("训练完成")
    logger.info(f"总训练时间: {total_train_time:.2f}秒, 平均每轮训练时间: {avg_epoch_time:.2f}秒")
    logger.info(
        f"最佳验证指标 (Epoch {best_epoch}), "
        f"RT MAE: {best_metrics['rt_mae']:.4f}, RT RMSE: {best_metrics['rt_rmse']:.4f}, "
        f"TP MAE: {best_metrics['tp_mae']:.4f}, TP RMSE: {best_metrics['tp_rmse']:.4f}, "
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_metrics


def evaluate_stem_model(model, valid_loader, device):
    """评估STEM模型性能（不进行反归一化）"""
    model.eval()
    criterion = nn.SmoothL1Loss(reduction='none')

    valid_loss = 0.0
    rt_losses = 0.0
    tp_losses = 0.0
    batch_count = 0

    rt_preds = []
    tp_preds = []
    rt_targets_list = []
    tp_targets_list = []

    with torch.no_grad():
        for batch in valid_loader:
            user_features = batch['user_info'].to(device)
            ws_features = batch['ws_info'].to(device)
            rt_targets = batch['rt'].to(device)
            tp_targets = batch['tp'].to(device)
            rt_mask = batch.get('rt_mask', torch.ones_like(rt_targets)).to(device)
            tp_mask = batch.get('tp_mask', torch.ones_like(tp_targets)).to(device)

            # 获取预测结果
            preds = model(user_features, ws_features)
            rt_pred = preds[0].squeeze(-1)
            tp_pred = preds[1].squeeze(-1)

            # 计算损失
            rt_per = criterion(rt_pred, rt_targets)
            tp_per = criterion(tp_pred, tp_targets)
            rt_denom = torch.clamp(rt_mask.sum(), min=1.0)
            tp_denom = torch.clamp(tp_mask.sum(), min=1.0)
            rt_loss = (rt_per * rt_mask).sum() / rt_denom
            tp_loss = (tp_per * tp_mask).sum() / tp_denom
            total_loss = rt_loss + tp_loss

            valid_loss += total_loss.item()
            rt_losses += rt_loss.item()
            tp_losses += tp_loss.item()
            batch_count += 1

            # 收集预测和目标
            if rt_mask.sum().item() > 0:
                rt_mask_bool = rt_mask > 0
                rt_preds.append(rt_pred[rt_mask_bool])
                rt_targets_list.append(rt_targets[rt_mask_bool])
            if tp_mask.sum().item() > 0:
                tp_mask_bool = tp_mask > 0
                tp_preds.append(tp_pred[tp_mask_bool])
                tp_targets_list.append(tp_targets[tp_mask_bool])

    # 合并所有批次的预测和目标
    rt_preds = torch.cat(rt_preds) if len(rt_preds) > 0 else torch.tensor([], device=device)
    tp_preds = torch.cat(tp_preds) if len(tp_preds) > 0 else torch.tensor([], device=device)
    rt_targets = torch.cat(rt_targets_list) if len(rt_targets_list) > 0 else torch.tensor([], device=device)
    tp_targets = torch.cat(tp_targets_list) if len(tp_targets_list) > 0 else torch.tensor([], device=device)

    # 计算指标（直接使用原始值，不进行反归一化）
    if rt_preds.numel() > 0:
        rt_mae = torch.mean(torch.abs(rt_preds - rt_targets)).item()
        rt_mse = torch.mean((rt_preds - rt_targets) ** 2).item()
        rt_rmse = torch.sqrt(torch.tensor(rt_mse)).item()
    else:
        rt_mae, rt_rmse = float('nan'), float('nan')

    if tp_preds.numel() > 0:
        tp_mae = torch.mean(torch.abs(tp_preds - tp_targets)).item()
        tp_mse = torch.mean((tp_preds - tp_targets) ** 2).item()
        tp_rmse = torch.sqrt(torch.tensor(tp_mse)).item()
    else:
        tp_mae, tp_rmse = float('nan'), float('nan')

    # 计算平均损失
    avg_valid_loss = valid_loss / batch_count if batch_count > 0 else 0
    avg_rt_loss = rt_losses / batch_count if batch_count > 0 else 0
    avg_tp_loss = tp_losses / batch_count if batch_count > 0 else 0

    return {
        'total_loss': avg_valid_loss,
        'rt_loss': avg_rt_loss,
        'tp_loss': avg_tp_loss,
        'rt_mae': rt_mae,
        'rt_rmse': rt_rmse,
        'tp_mae': tp_mae,
        'tp_rmse': tp_rmse
    }


def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"已设置随机种子: {seed}")


def str2bool(value):
    """Parse common command-line boolean strings."""
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='STEM模型用于QoS预测')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/RTTP/')
    parser.add_argument('--userlist_path', type=str, default='data/RTTP/userlist.csv')
    parser.add_argument('--wslist_path', type=str, default='data/RTTP/wslist.csv')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--density', type=int, default=1, help='数据密度百分比')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)

    # STEM模型参数
    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_shared_experts', type=int, default=2)
    parser.add_argument('--num_specific_experts', type=int, default=2)
    parser.add_argument('--expert_hidden_dims', type=str, default='128,64')
    parser.add_argument('--gate_hidden_dims', type=str, default='64,32')
    parser.add_argument('--tower_hidden_dims', type=str, default='64,32')
    parser.add_argument('--batch_norm', default=False)
    parser.add_argument('--hidden_activations', type=str, default='relu')
    parser.add_argument('--dropout_rates', type=float, default=0.3)

    # 可选功能参数
    parser.add_argument('--use_task_specific_embedding', type=str2bool, nargs='?', const=True, default=True,
                        help='是否使用任务特定嵌入（差异化嵌入）。若为False，所有任务均使用共享嵌入')
    parser.add_argument('--use_stop_gradient', type=str2bool, nargs='?', const=True, default=True,
                        help='是否使用梯度停止门控机制。若为False，门控对所有专家的输出进行聚合，不执行停止梯度操作')
    parser.add_argument('--use_gradnorm', type=str2bool, nargs='?', const=True, default=True,
                        help='是否使用GradNorm损失加权。若为False，两个任务的损失直接相加')
    
    # GradNorm超参数
    parser.add_argument('--gradnorm_alpha', type=float, default=0.0,
                        help='GradNorm的alpha超参数，控制任务平衡的强度（默认1.5）')
    parser.add_argument('--gradnorm_lr', type=float, default=0.03,
                        help='GradNorm权重优化器的学习率（默认0.025）')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 解析维度参数
    expert_hidden_dims = [int(x) for x in args.expert_hidden_dims.split(',')]
    gate_hidden_dims = [int(x) for x in args.gate_hidden_dims.split(',')]
    tower_hidden_dims = [int(x) for x in args.tower_hidden_dims.split(',')]

    # 检查GPU可用性
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"正在加载STEM数据，密度: {args.density}%...")
    dataset = Dataset(
        density=args.density,
        userlist_path=args.userlist_path,
        wslist_path=args.wslist_path,
        data_dir=args.data_dir
    )

    # 特征信息字典
    user_feature_info = {
        'user_id': dataset.user_id_num,
        'user_country': dataset.user_country_num,
        'user_as': dataset.user_as_num,
        'user_latitude': dataset.user_latitude_num,
        'user_longitude': dataset.user_longitude_num,
    }

    ws_feature_info = {
        'ws_id': dataset.ws_id_num,
        'ws_wsdl_address': dataset.ws_wsdl_address_num,
        'ws_provider': dataset.ws_provider_num,
        'ws_ip_address': dataset.ws_ip_address_num,
        'ws_country': dataset.ws_country_num,
        'ws_as': dataset.ws_as_num,
        'ws_latitude': dataset.ws_latitude_num,
        'ws_longitude': dataset.ws_longitude_num,
    }

    # 创建PyTorch数据集
    train_torch_dataset = TorchDataset(dataset.train_dataset)
    valid_torch_dataset = TorchDataset(dataset.val_dataset)
    test_torch_dataset = TorchDataset(dataset.test_dataset)

    # 创建数据加载器
    train_loader = DataLoader(
        train_torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info("===== STEM模型配置 =====")
    logger.info(f"任务数: {args.num_tasks}")
    logger.info(f"嵌入维度: {args.embedding_dim}")
    logger.info(f"层数: {args.num_layers}")
    logger.info(f"共享专家数量: {args.num_shared_experts}")
    logger.info(f"任务特定专家数量: {args.num_specific_experts}")
    logger.info(f"专家隐藏层维度: {expert_hidden_dims}")
    logger.info(f"门控隐藏层维度: {gate_hidden_dims}")
    logger.info(f"任务塔隐藏层维度: {tower_hidden_dims}")
    logger.info(f"是否使用BatchNorm: {args.batch_norm}")
    logger.info(f"隐藏层激活函数: {args.hidden_activations}")
    logger.info(f"Dropout率: {args.dropout_rates}")
    logger.info(f"是否使用差异化嵌入: {args.use_task_specific_embedding}")
    logger.info(f"是否使用梯度停止门控: {args.use_stop_gradient}")
    logger.info(f"是否使用GradNorm: {args.use_gradnorm}")
    if args.use_gradnorm:
        logger.info(f"  - GradNorm Alpha: {args.gradnorm_alpha}")
        logger.info(f"  - GradNorm 学习率: {args.gradnorm_lr}")
    logger.info("========================")

    # 创建STEM模型
    model = STEM(
        user_feature_info=user_feature_info,
        ws_feature_info=ws_feature_info,
        num_tasks=args.num_tasks,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_shared_experts=args.num_shared_experts,
        num_specific_experts=args.num_specific_experts,
        expert_hidden_units=expert_hidden_dims,
        gate_hidden_units=gate_hidden_dims,
        tower_hidden_units=tower_hidden_dims,
        batch_norm=args.batch_norm,
        hidden_activations=args.hidden_activations,
        dropout_rates=args.dropout_rates,
        use_task_specific_embedding=args.use_task_specific_embedding,
        use_stop_gradient=args.use_stop_gradient,
        use_gradnorm=args.use_gradnorm
    ).to(device)

    # 统计模型参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数数量: {trainable_params:,}")

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    # 训练STEM模型
    best_val_metrics = train_stem_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        eval_epochs=args.eval_epochs,
        early_stopping_patience=args.early_stopping_patience,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr
    )

    # 在测试集上评估，将测试性能作为最终性能
    test_metrics = evaluate_stem_model(
        model=model,
        valid_loader=test_loader,
        device=device
    )

    logger.info(
        f"最终测试指标: "
        f"Loss: {test_metrics['total_loss']:.4f}, "
        f"RT MAE: {test_metrics['rt_mae']:.4f}, RT RMSE: {test_metrics['rt_rmse']:.4f}, "
        f"TP MAE: {test_metrics['tp_mae']:.4f}, TP RMSE: {test_metrics['tp_rmse']:.4f}"
    )

    return test_metrics


if __name__ == '__main__':
    main()

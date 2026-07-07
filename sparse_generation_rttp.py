from pathlib import Path
import pandas as pd
import random
import numpy as np
import torch

# 设置随机种子
random.seed(2025)
np.random.seed(2025)

# 获取rtMatrix存放路径
rtMatrix_csv = "data/RTTP/rtMatrix.csv"
# 读取rtMatrix
rtMatrix_df = pd.read_csv(rtMatrix_csv, header=None)
# 保持 -1 不变（表示没有 QoS 记录）
# df转换为数组形式
rtMatrix_array = rtMatrix_df.values

# 获取tpMatrix存放路径
tpMatrix_csv = "data/RTTP/tpMatrix.csv"
# 读取 tpMatrix
tpMatrix_df = pd.read_csv(tpMatrix_csv, header=None)
# 保持 -1 不变（表示没有 QoS 记录）
# df转换为数组形式
tpMatrix_array = tpMatrix_df.values

# 不再需要两个矩阵都不为零的过滤条件
# 保留原始矩阵的数据，不做交集处理

# 获取稀疏矩阵存放目录（训练集、验证集、测试集）
def get_file_paths(density: float):
    """根据密度返回对应的文件路径"""
    base_path = "data/RTTP/sparse"
    return {
        'rt_train': f"{base_path}/rtMatrix{density}_train.csv",
        'rt_val': f"{base_path}/rtMatrix{density}_val.csv",
        'rt_test': f"{base_path}/rtMatrix{density}_test.csv",
        'tp_train': f"{base_path}/tpMatrix{density}_train.csv",
        'tp_val': f"{base_path}/tpMatrix{density}_val.csv",
        'tp_test': f"{base_path}/tpMatrix{density}_test.csv",
    }


def Generate_Sparse_Matrix(density: float):
    """
    按给定矩阵密度划分训练集、验证集和测试集。

    - 先在原始 RT / TP 矩阵上找出所有"有效交互"（rt!=0 或 tp!=0）。
      每个有效交互可以只包含一个 QoS（只 RT 或只 TP），也可以两个 QoS 都有。
      不会包含"两个 QoS 都为 0"的无效交互。
    - 随机选择 density% × 80% 的有效交互作为训练集
    - 再选择 density% × 20% 的有效交互作为验证集
    - 其余有效交互全部作为测试集

    参数:
    density: 密度百分比，例如 1 表示随机选择 1% × 80% = 0.8% 作为训练集，1% × 20% = 0.2% 作为验证集，99% 作为测试集。
    """
    # 获取原始矩阵的形状（行数和列数）
    rows, cols = rtMatrix_array.shape

    # 创建一个和原始矩阵同样形状的矩阵，用来存储分割后的训练、验证、测试数据
    # 初始化为 -1（表示没有 QoS 记录）
    rtMatrix_train_array = np.full_like(rtMatrix_array, -1.0)
    rtMatrix_val_array = np.full_like(rtMatrix_array, -1.0)
    rtMatrix_test_array = np.full_like(rtMatrix_array, -1.0)
    tpMatrix_train_array = np.full_like(tpMatrix_array, -1.0)
    tpMatrix_val_array = np.full_like(tpMatrix_array, -1.0)
    tpMatrix_test_array = np.full_like(tpMatrix_array, -1.0)

    # 找到所有"有效交互"：至少有一个 QoS 非零（rt != -1 或 tp != -1）
    valid_indices = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if (rtMatrix_array[i, j] != -1) or (tpMatrix_array[i, j] != -1)
    ]

    # 如果没有有效交互，直接返回空矩阵
    if not valid_indices:
        print("没有找到任何有效交互（RT 和 TP 全为 0），不生成稀疏矩阵。")
        return

    # 打乱索引顺序
    random.shuffle(valid_indices)

    # 计算训练集、验证集中应包含的"交互记录"数量
    # 训练集：density% × 80%
    train_size = int(density / 100.0 * 0.8 * len(valid_indices))
    train_size = max(1, train_size)  # 至少保留 1 条
    
    # 验证集：density% × 20%
    val_size = int(density / 100.0 * 0.2 * len(valid_indices))
    val_size = max(1, val_size)  # 至少保留 1 条

    # 划分索引
    train_indices = set(valid_indices[:train_size])
    val_indices = set(valid_indices[train_size:train_size + val_size])
    test_indices = set(valid_indices[train_size + val_size:])

    # 填充训练矩阵
    for i, j in train_indices:
        if rtMatrix_array[i, j] != -1:
            rtMatrix_train_array[i, j] = rtMatrix_array[i, j]
        if tpMatrix_array[i, j] != -1:
            tpMatrix_train_array[i, j] = tpMatrix_array[i, j]

    # 填充验证矩阵
    for i, j in val_indices:
        if rtMatrix_array[i, j] != -1:
            rtMatrix_val_array[i, j] = rtMatrix_array[i, j]
        if tpMatrix_array[i, j] != -1:
            tpMatrix_val_array[i, j] = tpMatrix_array[i, j]

    # 填充测试矩阵
    for i, j in test_indices:
        if rtMatrix_array[i, j] != -1:
            rtMatrix_test_array[i, j] = rtMatrix_array[i, j]
        if tpMatrix_array[i, j] != -1:
            tpMatrix_test_array[i, j] = tpMatrix_array[i, j]

    # 数组转DataFrame
    rtMatrix_train_df = pd.DataFrame(rtMatrix_train_array)
    rtMatrix_val_df = pd.DataFrame(rtMatrix_val_array)
    rtMatrix_test_df = pd.DataFrame(rtMatrix_test_array)
    tpMatrix_train_df = pd.DataFrame(tpMatrix_train_array)
    tpMatrix_val_df = pd.DataFrame(tpMatrix_val_array)
    tpMatrix_test_df = pd.DataFrame(tpMatrix_test_array)

    # 获取文件路径
    file_paths = get_file_paths(density)
    
    # 将DataFrame保存为CSV文件
    rtMatrix_train_df.to_csv(file_paths['rt_train'], index=False, header=False)
    rtMatrix_val_df.to_csv(file_paths['rt_val'], index=False, header=False)
    rtMatrix_test_df.to_csv(file_paths['rt_test'], index=False, header=False)
    tpMatrix_train_df.to_csv(file_paths['tp_train'], index=False, header=False)
    tpMatrix_val_df.to_csv(file_paths['tp_val'], index=False, header=False)
    tpMatrix_test_df.to_csv(file_paths['tp_test'], index=False, header=False)

    # 统计有效元素（不等于-1）
    rt_train_count = np.count_nonzero(rtMatrix_train_array != -1)
    rt_val_count = np.count_nonzero(rtMatrix_val_array != -1)
    rt_test_count = np.count_nonzero(rtMatrix_test_array != -1)
    tp_train_count = np.count_nonzero(tpMatrix_train_array != -1)
    tp_val_count = np.count_nonzero(tpMatrix_val_array != -1)
    tp_test_count = np.count_nonzero(tpMatrix_test_array != -1)

    # 计算训练集、验证集、测试集中 RT / TP 的交集及其占比
    train_overlap_count = 0
    val_overlap_count = 0
    test_overlap_count = 0

    # 训练集交集：同一位置上 RT 和 TP 都有有效值（!= -1）
    if rt_train_count > 0 and tp_train_count > 0:
        train_overlap_mask = (rtMatrix_train_array != -1) & (tpMatrix_train_array != -1)
        train_overlap_count = int(np.count_nonzero(train_overlap_mask))

    # 验证集交集：同一位置上 RT 和 TP 都有有效值（!= -1）
    if rt_val_count > 0 and tp_val_count > 0:
        val_overlap_mask = (rtMatrix_val_array != -1) & (tpMatrix_val_array != -1)
        val_overlap_count = int(np.count_nonzero(val_overlap_mask))

    # 测试集交集：同一位置上 RT 和 TP 都有有效值（!= -1）
    if rt_test_count > 0 and tp_test_count > 0:
        test_overlap_mask = (rtMatrix_test_array != -1) & (tpMatrix_test_array != -1)
        test_overlap_count = int(np.count_nonzero(test_overlap_mask))

    train_overlap_ratio = (
        train_overlap_count / min(rt_train_count, tp_train_count)
        if min(rt_train_count, tp_train_count) > 0
        else 0.0
    )
    val_overlap_ratio = (
        val_overlap_count / min(rt_val_count, tp_val_count)
        if min(rt_val_count, tp_val_count) > 0
        else 0.0
    )
    test_overlap_ratio = (
        test_overlap_count / min(rt_test_count, tp_test_count)
        if min(rt_test_count, tp_test_count) > 0
        else 0.0
    )

    # 打印统计信息
    total_valid = len(valid_indices)
    train_ratio = (len(train_indices) / total_valid * 100) if total_valid > 0 else 0
    val_ratio = (len(val_indices) / total_valid * 100) if total_valid > 0 else 0
    test_ratio = (len(test_indices) / total_valid * 100) if total_valid > 0 else 0
    
    print(f"密度参数: {density}%")
    print(f"总有效交互数: {total_valid}")
    print(f"训练集交互数: {len(train_indices)} ({train_ratio:.2f}% = {density}% × 80%)")
    print(f"验证集交互数: {len(val_indices)} ({val_ratio:.2f}% = {density}% × 20%)")
    print(f"测试集交互数: {len(test_indices)} ({test_ratio:.2f}%)")
    print(f"RT训练集元素数: {rt_train_count}")
    print(f"RT验证集元素数: {rt_val_count}")
    print(f"RT测试集元素数: {rt_test_count}")
    print(f"TP训练集元素数: {tp_train_count}")
    print(f"TP验证集元素数: {tp_val_count}")
    print(f"TP测试集元素数: {tp_test_count}")
    print(f"训练集交集元素数: {train_overlap_count}")
    print(f"验证集交集元素数: {val_overlap_count}")
    print(f"测试集交集元素数: {test_overlap_count}")
    print(f"训练集交集比例: {train_overlap_ratio:.4f}")
    print(f"验证集交集比例: {val_overlap_ratio:.4f}")
    print(f"测试集交集比例: {test_overlap_ratio:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    # 输出原始RT和TP矩阵的有效元素总数（!= -1）
    rt_total = np.count_nonzero(rtMatrix_array != -1)
    tp_total = np.count_nonzero(tpMatrix_array != -1)
    total_overlap = len(set([(i, j) for i in range(rtMatrix_array.shape[0]) for j in range(rtMatrix_array.shape[1])
                             if rtMatrix_array[i, j] != -1]).intersection(
        set([(i, j) for i in range(tpMatrix_array.shape[0]) for j in range(tpMatrix_array.shape[1])
             if tpMatrix_array[i, j] != -1])))

    print(f"原始RT矩阵有效元素总数: {rt_total}")
    print(f"原始TP矩阵有效元素总数: {tp_total}")
    print(f"原始RT和TP矩阵交集元素总数: {total_overlap}")
    print(f"原始交集占比: {total_overlap / min(rt_total, tp_total):.4f}")
    print("=" * 50)

    Generate_Sparse_Matrix(1)
    Generate_Sparse_Matrix(2)
    Generate_Sparse_Matrix(3)
    Generate_Sparse_Matrix(4)
    # Generate_Sparse_Matrix(5)
    # Generate_Sparse_Matrix(10)
    # Generate_Sparse_Matrix(15)
    # Generate_Sparse_Matrix(20)

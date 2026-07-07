"""
RTFP 数据集划分脚本
将完整的 RT 和 FP 矩阵按 8:1:1 的比例划分为训练集、验证集和测试集

输入：
- data/RTFP/rtMatrix.csv - 完整的响应时间矩阵
- data/RTFP/fpMatrix.csv - 完整的失败率矩阵

输出（保存到 data/RTFP/sparse/）：
- rtMatrix_train.csv - 训练集响应时间矩阵（80%）
- rtMatrix_val.csv - 验证集响应时间矩阵（10%）
- rtMatrix_test.csv - 测试集响应时间矩阵（10%）
- fpMatrix_train.csv - 训练集失败率矩阵（80%）
- fpMatrix_val.csv - 验证集失败率矩阵（10%）
- fpMatrix_test.csv - 测试集失败率矩阵（10%）

注：使用 -1 表示没有 QoS 记录，0 表示有记录但值为 0（如失败率为 0）
"""

import os
import numpy as np
import pandas as pd
import random
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置随机种子
random.seed(2025)
np.random.seed(2025)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("RTFP 数据集划分处理 (4:1:5)")
    print("=" * 80 + "\n")
    
    # ========== 1. 设置路径 ==========
    rtfp_dir = "data/RTFP"
    rt_matrix_path = os.path.join(rtfp_dir, "rtMatrix.csv")
    fp_matrix_path = os.path.join(rtfp_dir, "fpMatrix.csv")
    output_dir = os.path.join(rtfp_dir, "sparse")
    
    # 检查输入文件是否存在
    if not os.path.exists(rt_matrix_path):
        logger.error(f"RT 矩阵文件不存在: {rt_matrix_path}")
        return
    
    if not os.path.exists(fp_matrix_path):
        logger.error(f"FP 矩阵文件不存在: {fp_matrix_path}")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}\n")
    
    # ========== 2. 读取矩阵 ==========
    logger.info("读取矩阵文件...")
    
    rtMatrix_array = pd.read_csv(rt_matrix_path, header=None).values
    fpMatrix_array = pd.read_csv(fp_matrix_path, header=None).values
    
    logger.info(f"RT 矩阵形状: {rtMatrix_array.shape}")
    logger.info(f"FP 矩阵形状: {fpMatrix_array.shape}")
    
    # 检查矩阵形状是否一致
    if rtMatrix_array.shape != fpMatrix_array.shape:
        logger.error(f"RT 和 FP 矩阵形状不一致！")
        return
    
    # 输出原始矩阵统计信息
    total_elements = rtMatrix_array.shape[0] * rtMatrix_array.shape[1]
    logger.info(f"\n原始矩阵统计:")
    logger.info(f"总元素数: {total_elements}")
    logger.info(f"RT 矩阵有效元素: {total_elements} (100%)")
    logger.info(f"FP 矩阵有效元素: {total_elements} (100%)")
    logger.info(f"RT 和 FP 交集元素: {total_elements} (100%)")
    logger.info("=" * 80)
    
    # ========== 3. 划分数据集 ==========
    logger.info("\n划分数据集 (40% 训练, 10% 验证, 50% 测试)...")
    
    # 获取原始矩阵的形状
    rows, cols = rtMatrix_array.shape
    
    # 创建矩阵，初始化为 -1（表示没有 QoS 记录）
    rtMatrix_train_array = np.full_like(rtMatrix_array, -1.0)
    rtMatrix_val_array = np.full_like(rtMatrix_array, -1.0)
    rtMatrix_test_array = np.full_like(rtMatrix_array, -1.0)
    fpMatrix_train_array = np.full_like(fpMatrix_array, -1.0)
    fpMatrix_val_array = np.full_like(fpMatrix_array, -1.0)
    fpMatrix_test_array = np.full_like(fpMatrix_array, -1.0)
    
    # 创建所有位置的索引
    valid_indices = [(i, j) for i in range(rows) for j in range(cols)]
    
    # 打乱索引顺序
    random.shuffle(valid_indices)
    
    # 计算划分点
    total_count = len(valid_indices)
    train_size = int(total_count * 0.4)
    val_size = int(total_count * 0.1)
    
    # 划分索引
    train_indices = set(valid_indices[:train_size])
    val_indices = set(valid_indices[train_size:train_size + val_size])
    test_indices = set(valid_indices[train_size + val_size:])
    
    logger.info(f"总元素数: {total_count}")
    logger.info(f"训练集: {len(train_indices)} ({len(train_indices)/total_count*100:.2f}%)")
    logger.info(f"验证集: {len(val_indices)} ({len(val_indices)/total_count*100:.2f}%)")
    logger.info(f"测试集: {len(test_indices)} ({len(test_indices)/total_count*100:.2f}%)")
    
    # 填充训练矩阵
    for i, j in train_indices:
        rtMatrix_train_array[i, j] = rtMatrix_array[i, j]
        fpMatrix_train_array[i, j] = fpMatrix_array[i, j]
    
    # 填充验证矩阵
    for i, j in val_indices:
        rtMatrix_val_array[i, j] = rtMatrix_array[i, j]
        fpMatrix_val_array[i, j] = fpMatrix_array[i, j]
    
    # 填充测试矩阵
    for i, j in test_indices:
        rtMatrix_test_array[i, j] = rtMatrix_array[i, j]
        fpMatrix_test_array[i, j] = fpMatrix_array[i, j]
    
    # ========== 4. 保存矩阵 ==========
    logger.info("\n保存划分后的矩阵...")
    
    # 保存 RT 矩阵
    rt_train_path = os.path.join(output_dir, "rtMatrix_train.csv")
    rt_val_path = os.path.join(output_dir, "rtMatrix_val.csv")
    rt_test_path = os.path.join(output_dir, "rtMatrix_test.csv")
    
    pd.DataFrame(rtMatrix_train_array).to_csv(rt_train_path, index=False, header=False)
    pd.DataFrame(rtMatrix_val_array).to_csv(rt_val_path, index=False, header=False)
    pd.DataFrame(rtMatrix_test_array).to_csv(rt_test_path, index=False, header=False)
    
    logger.info(f"  ✓ RT 训练集: {rt_train_path}")
    logger.info(f"  ✓ RT 验证集: {rt_val_path}")
    logger.info(f"  ✓ RT 测试集: {rt_test_path}")
    
    # 保存 FP 矩阵
    fp_train_path = os.path.join(output_dir, "fpMatrix_train.csv")
    fp_val_path = os.path.join(output_dir, "fpMatrix_val.csv")
    fp_test_path = os.path.join(output_dir, "fpMatrix_test.csv")
    
    pd.DataFrame(fpMatrix_train_array).to_csv(fp_train_path, index=False, header=False)
    pd.DataFrame(fpMatrix_val_array).to_csv(fp_val_path, index=False, header=False)
    pd.DataFrame(fpMatrix_test_array).to_csv(fp_test_path, index=False, header=False)
    
    logger.info(f"  ✓ FP 训练集: {fp_train_path}")
    logger.info(f"  ✓ FP 验证集: {fp_val_path}")
    logger.info(f"  ✓ FP 测试集: {fp_test_path}")
    
    # ========== 5. 统计信息 ==========
    rt_train_count = np.count_nonzero(rtMatrix_train_array != -1)
    rt_val_count = np.count_nonzero(rtMatrix_val_array != -1)
    rt_test_count = np.count_nonzero(rtMatrix_test_array != -1)
    fp_train_count = np.count_nonzero(fpMatrix_train_array != -1)
    fp_val_count = np.count_nonzero(fpMatrix_val_array != -1)
    fp_test_count = np.count_nonzero(fpMatrix_test_array != -1)
    
    train_overlap_count = np.count_nonzero(
        (rtMatrix_train_array != -1) & (fpMatrix_train_array != -1)
    )
    val_overlap_count = np.count_nonzero(
        (rtMatrix_val_array != -1) & (fpMatrix_val_array != -1)
    )
    test_overlap_count = np.count_nonzero(
        (rtMatrix_test_array != -1) & (fpMatrix_test_array != -1)
    )
    
    logger.info(f"\n统计信息:")
    logger.info(f"RT 训练集元素数: {rt_train_count}")
    logger.info(f"RT 验证集元素数: {rt_val_count}")
    logger.info(f"RT 测试集元素数: {rt_test_count}")
    logger.info(f"FP 训练集元素数: {fp_train_count}")
    logger.info(f"FP 验证集元素数: {fp_val_count}")
    logger.info(f"FP 测试集元素数: {fp_test_count}")
    logger.info(f"训练集交集元素数: {train_overlap_count}")
    logger.info(f"验证集交集元素数: {val_overlap_count}")
    logger.info(f"测试集交集元素数: {test_overlap_count}")
    
    # ========== 6. 完成 ==========
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    print("  训练集 (40%):")
    print(f"    - rtMatrix_train.csv")
    print(f"    - fpMatrix_train.csv")
    print("  验证集 (10%):")
    print(f"    - rtMatrix_val.csv")
    print(f"    - fpMatrix_val.csv")
    print("  测试集 (50%):")
    print(f"    - rtMatrix_test.csv")
    print(f"    - fpMatrix_test.csv")
    print("\n注：未被选中的位置填充为 -1（表示没有 QoS 记录）")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按回车键退出...")

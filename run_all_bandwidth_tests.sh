#!/bin/bash
# 完整的内存带宽测试脚本：单机 + 多机

echo "========================================="
echo "内存带宽完整测试套件"
echo "========================================="
echo ""

# ==================== 测试1: 单机OpenMP带宽 ====================
echo "========================================="
echo "测试1: 单机OpenMP内存带宽"
echo "========================================="
echo ""

bash test_bandwidth.sh

echo ""
echo "========================================="
echo "单机测试完成"
echo "========================================="
echo ""

# ==================== 测试2: MPI多机带宽 ====================
echo ""
echo "========================================="
echo "测试2: MPI多机内存带宽"
echo "========================================="
echo ""

bash test_bandwidth_mpi.sh

echo ""
echo "========================================="
echo "多机测试完成"
echo "========================================="
echo ""

# ==================== 生成综合报告 ====================
echo ""
echo "========================================="
echo "生成综合报告..."
echo "========================================="
echo ""

python3 << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建综合图表
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 图1: 单机OpenMP带宽扩展
ax1 = fig.add_subplot(gs[0, 0])
omp_data = pd.read_csv('bandwidth_results.csv')
threads = omp_data['线程数']
avg_bw = omp_data[' 平均(GB/s)']

ax1.plot(threads, avg_bw, 'o-', linewidth=3, markersize=10, color='#2E86AB')
ax1.fill_between(threads, 0, avg_bw, alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax1.set_title('Single Node: OpenMP Bandwidth Scaling', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(threads)

# 标注峰值
max_idx = avg_bw.idxmax()
max_thr = threads[max_idx]
max_bw = avg_bw[max_idx]
ax1.annotate(f'Peak: {max_bw:.1f} GB/s @ {max_thr} threads',
            xy=(max_thr, max_bw),
            xytext=(max_thr+3, max_bw-5),
            fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# 图2: MPI聚合带宽（不同OMP配置）
ax2 = fig.add_subplot(gs[0, 1])
mpi_data = pd.read_csv('bandwidth_mpi_results.csv')
omp_configs = mpi_data['OMP_Per_Rank'].unique()
markers = ['o', 's', '^', 'd']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, omp in enumerate(omp_configs):
    data = mpi_data[mpi_data['OMP_Per_Rank'] == omp]
    ax2.plot(data['MPI_Ranks'], data['Aggregate_BW(GB/s)'],
            marker=markers[i], linewidth=2.5, markersize=9,
            label=f'{omp} OMP/rank', color=colors[i])

ax2.set_xlabel('MPI Ranks', fontsize=13, fontweight='bold')
ax2.set_ylabel('Aggregate Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax2.set_title('Multi-Node: MPI Aggregate Bandwidth', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, title='OMP per Rank')
ax2.grid(True, alpha=0.3)

# 图3: 对比图 - 单机 vs 多机
ax3 = fig.add_subplot(gs[1, :])

# 单机带宽（作为1个MPI rank）
single_node_max = avg_bw.max()
ax3.bar(0, single_node_max, width=0.5, label='Single Node (OpenMP)',
        color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=2)

# 多机带宽
mpi_ranks = mpi_data['MPI_Ranks'].unique()
for i, ranks in enumerate(mpi_ranks):
    data = mpi_data[mpi_data['MPI_Ranks'] == ranks]
    # 取每个MPI配置下的最大带宽
    max_bw = data['Aggregate_BW(GB/s)'].max()
    ax3.bar(ranks, max_bw, width=0.5,
            label=f'{ranks} MPI Ranks' if i == 0 else '',
            color='#A23B72', alpha=0.6 + i*0.1, edgecolor='black', linewidth=1.5)

ax3.set_xlabel('Number of MPI Ranks (0 = Single Node)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Aggregate Memory Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax3.set_title('Single Node vs Multi-Node: Aggregate Bandwidth Comparison',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks([0] + list(mpi_ranks))
ax3.set_xticklabels(['Single\nNode'] + [str(x) for x in mpi_ranks])

# 添加数值标注
ax3.text(0, single_node_max + 5, f'{single_node_max:.1f}',
        ha='center', fontsize=11, fontweight='bold')
for i, ranks in enumerate(mpi_ranks):
    data = mpi_data[mpi_data['MPI_Ranks'] == ranks]
    max_bw = data['Aggregate_BW(GB/s)'].max()
    ax3.text(ranks, max_bw + 5, f'{max_bw:.1f}',
            ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Memory Bandwidth Analysis: Single Node vs Multi-Node',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('bandwidth_comprehensive_report.png', dpi=300, bbox_inches='tight')
print("综合报告图已保存: bandwidth_comprehensive_report.png")

# 生成文本报告
print("\n" + "="*60)
print("内存带宽测试总结报告")
print("="*60)

print("\n【单机OpenMP带宽】")
print(f"  峰值带宽: {single_node_max:.2f} GB/s @ {max_thr} threads")
print(f"  1线程带宽: {avg_bw.iloc[0]:.2f} GB/s")
print(f"  扩展比: {single_node_max / avg_bw.iloc[0]:.2f}x")

print("\n【MPI多机聚合带宽】")
max_mpi = mpi_data['Aggregate_BW(GB/s)'].max()
max_config = mpi_data.loc[mpi_data['Aggregate_BW(GB/s)'].idxmax()]
print(f"  峰值聚合带宽: {max_mpi:.2f} GB/s")
print(f"  配置: {max_config['MPI_Ranks']} MPI × {max_config['OMP_Per_Rank']} OMP = {max_config['Total_Cores']} 核")
print(f"  相对单机提升: {max_mpi / single_node_max:.2f}x")

single_rank_avg = mpi_data[mpi_data['MPI_Ranks'] == 1]['Aggregate_BW(GB/s)'].iloc[0]
print(f"  单个rank平均带宽: {single_rank_avg:.2f} GB/s")

print("\n【关键发现】")
if max_mpi > single_node_max:
    print(f"  ✓ MPI多机扩展有效，聚合带宽是单机的 {max_mpi / single_node_max:.2f}x")
else:
    print(f"  ✗ MPI多机扩展受限，可能原因：网络带宽/通信开销")

# 检查带宽饱和
if len(threads) > 1:
    bw_growth = avg_bw.iloc[-1] / avg_bw.iloc[0]
    thread_growth = threads.iloc[-1] / threads.iloc[0]
    if bw_growth < thread_growth * 0.5:
        print(f"  ✓ 单机带宽在{threads.iloc[-1]}线程时已饱和")

EOF

echo ""
echo "========================================="
echo "所有测试完成！"
echo "========================================="
echo ""
echo "生成的文件："
echo "  1. bandwidth_results.csv        - 单机测试数据"
echo "  2. bandwidth_mpi_results.csv    - 多机测试数据"
echo "  3. bandwidth_scaling.png        - 单机扩展性图"
echo "  4. bandwidth_mpi_scaling.png    - 多机扩展性图"
echo "  5. bandwidth_comprehensive_report.png - 综合报告图"
echo ""
echo "请将所有.png图片和.csv文件打包发送用于分析"

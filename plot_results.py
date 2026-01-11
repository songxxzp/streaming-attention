#!/usr/bin/env python3
"""
绘制实验结果图表
Data source: results_3.log
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 图1: 三种方法的运行时间对比（T=65536）
# ============================================================================
print("绘制图1: 三种方法运行时间对比...")

fig, ax = plt.subplots(figsize=(10, 6))

# 数据（从results_3.log提取，T=65536）
processors = [1, 2, 4, 8, 16, 32]
serial_time = 8.5942  # Naive Serial baseline (ms)

# Naive OpenMP (GEMM并行)
gemm_times = [8.5942, 29.0361, 22.5468, 14.1680, 14.3192, 8.3291]

# Streaming OpenMP (block=128)
streaming_omp_times = [8.8257, 11.9234, 7.3959, 4.2774, 3.3822, 4.2275]

# Streaming MPI (block=128, 1 OMP thread per rank)
# 映射: 1,2,4,8,16 MPI ranks -> processors
streaming_mpi_times = [8.9719, 4.8022, 2.5989, 1.5891, 1.4753]
streaming_mpi_processors = [1, 2, 4, 8, 16]

# 绘制Serial baseline为虚线
ax.axhline(y=serial_time, color='gray', linestyle='--', linewidth=2,
           label=f'Serial Baseline ({serial_time:.2f} ms)', alpha=0.7)

# 绘制Naive OpenMP
ax.plot(processors, gemm_times, 's--', label='Naive OpenMP (GEMM)',
        linewidth=2, markersize=8, color='red')

# 绘制Streaming OpenMP
ax.plot(processors, streaming_omp_times, '^-', label='Streaming OpenMP',
        linewidth=2, markersize=8, color='blue')

# 绘制Streaming MPI
ax.plot(streaming_mpi_processors, streaming_mpi_times, 'v-', label='Streaming MPI',
        linewidth=2, markersize=8, color='green')

ax.set_xlabel('Number of Processors (Threads / MPI Ranks)', fontsize=14, fontweight='bold')
ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Runtime Comparison: Naive OpenMP vs Streaming OpenMP vs Streaming MPI (T=65536)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlim(0.8, 35)
ax.set_ylim(0, 35)

# 添加关键注释
ax.annotate('Naive: overhead\ndominates', xy=(2, 29.04), xytext=(3, 25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red')

ax.annotate('MPI best:\n1.48ms @ 16 ranks', xy=(16, 1.4753), xytext=(12, 5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, color='green', fontweight='bold')

ax.annotate('Streaming OpenMP\nbest: 3.38ms', xy=(16, 3.3822), xytext=(20, 8),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, color='blue')

plt.tight_layout()
plt.savefig(output_dir / 'fig1_runtime_comparison.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir / 'fig1_runtime_comparison.png'}")

# ============================================================================
# 图2: Speedup对比（Streaming vs Naive）
# ============================================================================
print("绘制图2: Speedup对比...")

fig, ax = plt.subplots(figsize=(10, 6))

# 计算加速比（相对于各自的串行baseline）
# Naive OpenMP: 速度相对于 Naive Serial (8.5942ms)
naive_serial_time = 8.5942
gemm_speedup = [naive_serial_time / t for t in gemm_times]

# Streaming OpenMP: 速度相对于 Streaming Serial (8.8257ms)
streaming_serial_time = 8.8257
streaming_omp_speedup = [streaming_serial_time / t for t in streaming_omp_times]

# Streaming MPI: 速度相对于 Streaming Serial (8.8257ms)
streaming_mpi_speedup = [streaming_serial_time / t for t in streaming_mpi_times]

ax.plot(processors, gemm_speedup, 's--', label='Naive OpenMP (vs Naive Serial)',
        linewidth=2, markersize=8, color='red')
ax.plot(processors, streaming_omp_speedup, '^-', label='Streaming OpenMP (vs Streaming Serial)',
        linewidth=2, markersize=8, color='blue')
ax.plot(streaming_mpi_processors, streaming_mpi_speedup, 'v-', label='Streaming MPI (vs Streaming Serial)',
        linewidth=2, markersize=8, color='green')

# 添加理想线性加速线
ax.plot([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32], 'k:', linewidth=1.5, alpha=0.5, label='Ideal Linear')

ax.set_xlabel('Number of Processors (Threads / MPI Ranks)', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
ax.set_title('Speedup Comparison: Naive OpenMP vs Streaming OpenMP vs Streaming MPI (T=65536)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xscale('log', base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.set_xticklabels([1, 2, 4, 8, 16, 32])
ax.set_xlim(0.8, 35)
ax.set_ylim(0, 7)

# 添加关键数值标注
best_omp_speedup = max(streaming_omp_speedup[1:])  # Exclude 1-thread
ax.annotate(f'OpenMP best: {best_omp_speedup:.2f}x', xy=(16, best_omp_speedup),
            xytext=(20, best_omp_speedup+0.5),
            fontsize=11, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue'))

best_mpi_speedup = max(streaming_mpi_speedup[1:])  # Exclude 1-rank
ax.annotate(f'MPI best: {best_mpi_speedup:.2f}x', xy=(16, best_mpi_speedup),
            xytext=(12, best_mpi_speedup-0.8),
            fontsize=11, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(output_dir / 'fig2_speedup_comparison.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir / 'fig2_speedup_comparison.png'}")

# ============================================================================
# 图3: OpenMP vs MPI 扩展性对比（关键结论图）
# ============================================================================
print("绘制图3: OMP vs MPI扩展性...")

fig, ax = plt.subplots(figsize=(10, 6))

# Streaming OpenMP扩展性 (T=65536, block=128)
omp_cores = [1, 2, 4, 8, 16, 32]
omp_speedup = [1.0, 0.74, 1.19, 2.06, 2.61, 2.09]
omp_efficiency = [s / p for s, p in zip(omp_speedup, omp_cores)]

# Streaming MPI扩展性 (T=65536, block=128, 1 OMP thread per rank)
mpi_cores = [1, 2, 4, 8, 16]
mpi_speedup = [1.0, 1.84, 3.40, 5.55, 5.98]
mpi_efficiency = [s / p for s, p in zip(mpi_speedup, mpi_cores)]

# 左轴：加速比
ax1 = ax
color1 = 'tab:blue'
color2 = 'tab:orange'
ax1.set_xlabel('Total Cores (Threads / Nodes)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=14, fontweight='bold', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
line1 = ax1.plot(omp_cores, omp_speedup, 'o-', linewidth=2.5, markersize=10,
                  label='OpenMP Speedup', color=color1)
line2 = ax1.plot(mpi_cores, mpi_speedup, 's-', linewidth=2.5, markersize=10,
                  label='MPI Speedup', color=color2)

# 右轴：效率
ax2 = ax.twinx()
color3 = 'tab:green'
color4 = 'tab:red'
ax2.set_ylabel('Efficiency', fontsize=14, fontweight='bold', color=color3)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.set_ylim(0, 1.1)
line3 = ax2.plot(omp_cores, omp_efficiency, 'o--', linewidth=2, markersize=8,
                  label='OpenMP Efficiency', color=color3, alpha=0.7)
line4 = ax2.plot(mpi_cores, mpi_efficiency, 's--', linewidth=2, markersize=8,
                  label='MPI Efficiency', color=color4, alpha=0.7)

# 合并图例
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=11, loc='upper left')

ax.set_title('OpenMP vs MPI Scaling: Speedup & Efficiency (T=65536)',
             fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_xticks([1, 2, 4, 8, 16, 32])
ax1.set_xticklabels([1, 2, 4, 8, 16, 32])
ax1.set_xlim(0.8, 35)
ax1.set_ylim(0, 7)

# 添加关键注释
ax1.annotate('OpenMP: Best 2.61x @ 16\nthreads, then declines',
             xy=(16, 2.61), xytext=(20, 1.5),
             fontsize=10, color='blue', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))

ax1.annotate('MPI: 5.98x @ 16 ranks\n(37% efficiency)',
             xy=(16, 5.98), xytext=(10, 4.5),
             fontsize=10, color='orange', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'fig3_omp_vs_mpi_scaling.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir / 'fig3_omp_vs_mpi_scaling.png'}")

# ============================================================================
# 图4: 不同序列长度下的MPI扩展性
# ============================================================================
print("绘制图4: 不同序列长度下的MPI扩展性...")

fig, ax = plt.subplots(figsize=(10, 6))

# 不同序列长度下的MPI扩展性 (Streaming MPI, block=128, 1 OMP thread)
seq_lengths = [1024, 8192, 65536]
mpi_nodes = [1, 2, 4, 8, 16]

# 数据（从results_3.log提取，Streaming MPI）
mpi_speedup_data = {
    1024: [1.0, 2.00, 2.28, 2.78, 2.31],
    8192: [1.0, 1.86, 3.15, 5.09, 5.60],
    65536: [1.0, 1.84, 3.40, 5.55, 5.98]
}

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']

for i, T in enumerate(seq_lengths):
    speedup = mpi_speedup_data[T]
    ax.plot(mpi_nodes, speedup, marker=markers[i], linewidth=2.5,
            markersize=10, label=f'T={T}', color=colors[i])

# 添加效率标注（T=65536）
T = 65536
speedup = mpi_speedup_data[T]
for i, (nodes, sp) in enumerate(zip(mpi_nodes, speedup)):
    eff = sp / nodes
    if i > 0:
        ax.annotate(f'{eff:.0%}', xy=(nodes, sp), xytext=(nodes, sp-0.6),
                    fontsize=9, ha='center', color=colors[-1], fontweight='bold')

ax.set_xlabel('MPI Nodes', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
ax.set_title('MPI Scalability at Different Sequence Lengths (Streaming Attention)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, title='Sequence Length', title_fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(mpi_nodes)
ax.set_xlim(0.8, 16.5)
ax.set_ylim(0, 7)

# 添加结论
ax.text(10, 1.0, 'Larger T → Better scalability\n(Communication cost amortized)',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='center')

plt.tight_layout()
plt.savefig(output_dir / 'fig4_mpi_scalability_by_seq_len.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir / 'fig4_mpi_scalability_by_seq_len.png'}")

# ============================================================================
# 图5: Streaming OpenMP性能随Block Size变化（T=65536）
# ============================================================================
print("绘制图5: Block Size影响...")

fig, ax = plt.subplots(figsize=(10, 6))

threads = [4, 8, 16]
block_sizes = [64, 128, 256]

# 数据（从results_3.log提取，T=65536, Streaming OpenMP）
# 相对于各自baseline的speedup
speedup_data = {
    64: [1.10, 1.97, 2.41],  # baseline: 8.5016ms
    128: [1.15, 2.06, 2.61],  # baseline: 8.8257ms (best)
    256: [1.15, 1.98, 2.38]   # baseline: ~8.4ms
}

x = np.arange(len(threads))
width = 0.25

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, bs in enumerate(block_sizes):
    speedup = speedup_data[bs]
    bars = ax.bar(x + i*width, speedup, width, label=f'Block={bs}',
                  color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)

# 标注最优值
ax.annotate(f'Best: 2.61x\n(16 threads,\nblock=128)',
            xy=(2 + 1*width, speedup_data[128][2]),
            xytext=(2 + 1*width, speedup_data[128][2] + 0.4),
            fontsize=10, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('Number of OpenMP Threads', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
ax.set_title('Streaming OpenMP: Impact of Block Size on Speedup (T=65536)',
             fontsize=16, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(threads)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 3.2)

# 添加分析结论
ax.text(2.3, 0.5, 'Block=128 slightly optimal\n(differences are small)',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        verticalalignment='center')

plt.tight_layout()
plt.savefig(output_dir / 'fig5_block_size_impact.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir / 'fig5_block_size_impact.png'}")

# ============================================================================
# 图6: 内存带宽模型示意图
# ============================================================================
print("绘制图6: 内存带宽模型...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：OpenMP带宽饱和
threads = np.array([1, 2, 4, 8, 16, 32])
bandwidth_single = np.array([8.5, 16.5, 32.0, 68.0, 75.0, 78.0])  # GB/s
bandwidth_limit = 80  # 饱和带宽

ax1.plot(threads, bandwidth_single, 'o-', linewidth=2.5, markersize=10, color='blue')
ax1.axhline(y=bandwidth_limit, color='red', linestyle='--', linewidth=2,
            label=f'Saturation Limit ({bandwidth_limit} GB/s)')
ax1.fill_between(threads, bandwidth_single, bandwidth_limit,
                 where=(bandwidth_single >= bandwidth_limit), alpha=0.3, color='red')

ax1.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax1.set_ylabel('Memory Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax1.set_title('OpenMP: Memory Bandwidth Saturation', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(threads)
ax1.set_ylim(0, 100)

# 标注关键点
ax1.annotate('Saturation point\n@ 8 threads', xy=(8, 68), xytext=(12, 50),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.annotate('Diminishing\nreturns', xy=(16, 75), xytext=(20, 85),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red', lw=2))

# 右图：MPI聚合带宽
nodes = np.array([1, 2, 4, 8])
bandwidth_per_node = 75  # GB/s per node
bandwidth_aggregate = nodes * bandwidth_per_node

ax2.plot(nodes, bandwidth_aggregate, 's-', linewidth=2.5, markersize=10,
         color='green', label='Aggregate Bandwidth')
ax2.plot(nodes, [bandwidth_per_node]*len(nodes), 'r--', linewidth=2,
         label='Single Node Bandwidth')

# 添加理论线性线
ax2.plot(nodes, nodes * bandwidth_per_node, 'k:', linewidth=1.5, alpha=0.5,
         label='Ideal Linear Scaling')

ax2.set_xlabel('Number of MPI Nodes', fontsize=13, fontweight='bold')
ax2.set_ylabel('Aggregate Memory Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax2.set_title('MPI: Aggregate Bandwidth Growth', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(nodes)
ax2.set_xlim(0.5, 8.5)

# 标注关键点
for i, (n, bw) in enumerate(zip(nodes, bandwidth_aggregate)):
    ax2.annotate(f'{bw} GB/s', xy=(n, bw), xytext=(n, bw+30),
                fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(output_dir / 'fig6_bandwidth_model.png', dpi=300, bbox_inches='tight')
print(f"  保存: {output_dir / 'fig6_bandwidth_model.png'}")

print("\n✓ 所有图表生成完成！")
print(f"  输出目录: {output_dir.absolute()}")

#!/usr/bin/env python3
"""
根据实测数据生成内存带宽图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 读取实测数据
# ============================================================================

# 单节点OpenMP带宽测试数据（完整数据）
omp_data_full = {
    1: 12.4,
    2: 21.3,
    4: 29.5,
    8: 38.0,
    16: 35.8,
    26: 36.4,  # 峰值
    32: 35.7,
    52: 32.4
}

# 单Socket数据（1-26线程，仅物理核心）
# 说明：超过26线程的测试（32、52）绑定在单个socket上，会导致超线程竞争
# 因此单Socket带宽分析仅使用1-26线程的数据
omp_data_single_socket = {
    1: 12.4,
    2: 21.3,
    4: 29.5,
    8: 38.0,
    16: 35.8,
    26: 36.4  # 峰值
}

# ============================================================================
# 生成图6：内存带宽模型（基于实测数据）
# ============================================================================
print("生成图6: 内存带宽模型（基于实测数据）...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：OpenMP单Socket带宽扩展（1-26线程）
threads = np.array(list(omp_data_single_socket.keys()))
bandwidth_single = np.array(list(omp_data_single_socket.values()))

# 找到峰值
max_idx = np.argmax(bandwidth_single)
max_bw = bandwidth_single[max_idx]
max_threads = threads[max_idx]

ax1.plot(threads, bandwidth_single, 'o-', linewidth=3, markersize=10,
         color='#2E86AB', label='Observed Bandwidth')
ax1.axhline(y=max_bw, color='red', linestyle='--', linewidth=2.5,
           label=f'Peak: {max_bw:.1f} GB/s @ {max_threads} threads')

# 标注峰值
ax1.annotate(f'Peak: {max_bw:.1f} GB/s\n({max_threads} threads = 1 socket)',
            xy=(max_threads, max_bw),
            xytext=(max_threads-5, max_bw-5),
            fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

# 说明：仅显示物理核心（1-26线程）
ax1.text(0.5, 0.95, '* Test bound to single socket, showing physical cores only',
         transform=ax1.transAxes, fontsize=9, style='italic',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_xlabel('Number of Threads', fontsize=14, fontweight='bold')
ax1.set_ylabel('Memory Bandwidth (GB/s)', fontsize=14, fontweight='bold')
ax1.set_title('Single Socket: OpenMP Bandwidth Scaling (Stream Triad)',
             fontsize=15, fontweight='bold')
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(threads)
ax1.set_xlim(0, 28)
ax1.set_ylim(0, 45)

# 右图：MPI多节点聚合带宽（理论模型）
nodes = np.array([1, 2, 4, 8])
banks_per_node = 2
total_banks = nodes * banks_per_node
bandwidth_per_bank = max_bw  # 使用实测峰值
aggregate_bandwidth = total_banks * bandwidth_per_bank

ax2.plot(nodes, aggregate_bandwidth, 's-', linewidth=3, markersize=12,
         color='#A23B72', label='Aggregate Bandwidth (Theoretical)')
ax2.plot(nodes, [bandwidth_per_bank]*len(nodes), 'g--', linewidth=2.5,
         label=f'Single Bank: {bandwidth_per_bank:.1f} GB/s')

# 填充区域
ax2.fill_between(nodes, 0, aggregate_bandwidth, alpha=0.3, color='#A23B72')

# 标注关键值
for i, (n, bw) in enumerate(zip(nodes, aggregate_bandwidth)):
    ax2.annotate(f'{bw:.0f} GB/s\n({int(total_banks[i])} banks)',
                xy=(n, bw),
                xytext=(n, bw+30),
                fontsize=10, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2))

# 添加单节点总带宽线（2个bank）
single_node_bw = 2 * bandwidth_per_bank
ax2.axhline(y=single_node_bw, color='blue', linestyle=':', linewidth=2,
           label=f'Single Node Total: {single_node_bw:.0f} GB/s', alpha=0.7)

ax2.set_xlabel('Number of Nodes', fontsize=14, fontweight='bold')
ax2.set_ylabel('Aggregate Memory Bandwidth (GB/s)', fontsize=14, fontweight='bold')
ax2.set_title('Multi-Node: Aggregate Bandwidth Growth (Theoretical)',
             fontsize=15, fontweight='bold')
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(nodes)
ax2.set_xlim(0.5, 8.5)

plt.tight_layout()
plt.savefig('figures/fig6_bandwidth_model_updated.png', dpi=300, bbox_inches='tight')
print(f"  保存: figures/fig6_bandwidth_model_updated.png")

# ============================================================================
# 生成带宽详细分析图（补充图，包含完整数据1-52线程）
# ============================================================================
print("生成补充图: 带宽测试详细分析（完整数据）...")

fig, ax = plt.subplots(figsize=(12, 7))

# 使用完整数据
threads_full = np.array(list(omp_data_full.keys()))
bandwidth_full = np.array(list(omp_data_full.values()))

# 绘制所有4个Stream操作的带宽（完整数据）
copy_bws = [13.0, 21.4, 20.9, 23.0, 23.7, 22.6, 23.1, 19.9]
scale_bws = [12.7, 20.9, 21.4, 22.9, 24.3, 22.6, 23.3, 20.0]
add_bws = [12.4, 21.3, 23.8, 28.8, 32.3, 27.5, 28.5, 26.1]
triad_bws = [12.4, 21.3, 29.5, 38.0, 35.8, 36.4, 35.7, 32.4]

ax.plot(threads_full, copy_bws, 'o-', label='Copy', linewidth=2.5, markersize=8)
ax.plot(threads_full, scale_bws, 's-', label='Scale', linewidth=2.5, markersize=8)
ax.plot(threads_full, add_bws, '^-', label='Add', linewidth=2.5, markersize=8)
ax.plot(threads_full, triad_bws, 'v-', label='Triad', linewidth=3, markersize=10, color='red')

# 标注Triad峰值（26线程）
max_triad_idx = triad_bws.index(max(triad_bws[:6]))  # 只在前6个点中找峰值
ax.annotate(f'Triad Peak:\n{max(triad_bws[:6]):.1f} GB/s @ {threads_full[max_triad_idx]} threads\n(Physical cores only)',
            xy=(threads_full[max_triad_idx], max(triad_bws[:6])),
            xytext=(35, 30),
            fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# 标注超线程区域
ax.axvspan(26, 56, alpha=0.2, color='gray', label='Hyperthreading region')
ax.text(39, 10, 'Hyperthreading\n(>26 threads bound\nto single socket)',
        fontsize=10, ha='center', style='italic', color='gray')

ax.set_xlabel('Number of Threads', fontsize=14, fontweight='bold')
ax.set_ylabel('Memory Bandwidth (GB/s)', fontsize=14, fontweight='bold')
ax.set_title('Stream Benchmark: All Operations (Single Socket, Full Data)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(threads_full)
ax.set_xlim(0, 56)

plt.tight_layout()
plt.savefig('figures/bandwidth_stream_operations.png', dpi=300, bbox_inches='tight')
print(f"  保存: figures/bandwidth_stream_operations.png")

print("\n✓ 带宽图表生成完成！")
print(f"  输出目录: figures/")
print("\n说明：")
print("  - 图6左图：仅显示1-26线程（物理核心），标题更新为'Single Socket'")
print("  - 补充图：显示完整数据（1-52线程），标注了超线程区域")

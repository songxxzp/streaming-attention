#!/bin/bash
# MPI配置诊断脚本

echo "========================================="
echo "MPI配置诊断"
echo "========================================="
echo ""

# 1. 检查MPI
echo "1. MPI配置检查"
echo "----------------------------------------"
which mpirun
mpirun --version | head -3
echo ""

# 2. 检查节点信息
echo "2. 节点信息"
echo "----------------------------------------"
echo "主机名: $(hostname)"
echo "CPU核心数: $(lscpu | grep "^CPU(s):" | awk '{print $2}')"
echo "NUMA节点数: $(lscpu | grep "NUMA node(s):" | awk '{print $3}')"
echo "内存: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# 3. 检查hosts文件
echo "3. Hosts文件检查"
echo "----------------------------------------"
if [ -f "hosts" ]; then
    echo "找到hosts文件，内容："
    cat hosts
    echo ""
    NUM_HOSTS=$(wc -l < hosts)
    echo "总共 $NUM_HOSTS 个节点"
else
    echo "未找到hosts文件"
    echo ""
    echo "检测其他可能的节点："
    # 检查/etc/hosts
    if grep -q "node" /etc/hosts; then
        echo "/etc/hosts中的节点："
        grep "node" /etc/hosts | grep -v "^#" | awk '{print $2, $1}'
    fi
fi
echo ""

# 4. 测试MPI分布
echo "4. MPI节点分布测试"
echo "----------------------------------------"

# 创建测试程序
cat > test_mpi_distribution.cpp << 'EOF'
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[256];
    pid_t pid = getpid();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 255);

    // 收集所有hostname
    int len = size * 256;
    char* all_hostnames = new char[len];
    int* all_pids = new int[size];

    MPI_Gather(hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&pid, 1, MPI_INT, all_pids, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI Rank分布：\n");
        printf("总数: %d ranks\n\n", size);

        // 统计每个hostname上的rank数
        struct HostInfo {
            char hostname[256];
            int count;
            int ranks[16];
        } hosts[16];
        int num_hosts = 0;

        for (int i = 0; i < size; i++) {
            char* h = all_hostnames + i * 256;
            int found = 0;
            for (int j = 0; j < num_hosts; j++) {
                if (strcmp(hosts[j].hostname, h) == 0) {
                    hosts[j].ranks[hosts[j].count++] = i;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                strcpy(hosts[num_hosts].hostname, h);
                hosts[num_hosts].count = 1;
                hosts[num_hosts].ranks[0] = i;
                num_hosts++;
            }
        }

        printf("节点分布统计：\n");
        for (int i = 0; i < num_hosts; i++) {
            printf("  %s: %d ranks (", hosts[i].hostname, hosts[i].count);
            for (int j = 0; j < hosts[i].count; j++) {
                printf("%d", hosts[i].ranks[j]);
                if (j < hosts[i].count - 1) printf(", ");
            }
            printf(")\n");
        }

        printf("\n详细分布：\n");
        for (int i = 0; i < size; i++) {
            printf("  Rank %2d: %s (PID %d)\n", i, all_hostnames + i * 256, all_pids[i]);
        }

        // 判断是否有问题
        printf("\n诊断结果：\n");
        if (num_hosts == 1) {
            printf("  ✗ 问题：所有MPI ranks都在同一个节点上！\n");
            printf("  建议：创建hosts文件并使用 --map-by ppr:1:node\n");
        } else if (num_hosts == size) {
            printf("  ✓ 正确：每个rank在不同的节点上\n");
        } else {
            printf("  ⚠ 部分共享：%d个节点承载%d个ranks\n", num_hosts, size);
        }

        delete[] all_hostnames;
        delete[] all_pids;
    }

    MPI_Finalize();
    return 0;
}
EOF

mpicxx test_mpi_distribution.cpp -o test_mpi_distribution

if [ -f "hosts" ]; then
    echo "测试2个rank（使用hosts文件）："
    mpirun -np 2 --hostfile hosts ./test_mpi_distribution
    echo ""
fi

echo "测试4个rank（自动分配）："
mpirun -np 4 ./test_mpi_distribution
echo ""

# 5. 建议配置
echo "========================================="
echo "配置建议"
echo "========================================="
echo ""

if [ ! -f "hosts" ]; then
    echo "⚠ 未找到hosts文件"
    echo ""
    echo "如果只有1个节点："
    echo "  - 无法测试真正的多机带宽"
    echo "  - 建议修改报告，使用定性分析而非具体带宽数值"
    echo ""
    echo "如果有多个节点："
    echo "  - 创建hosts文件，每行一个节点名"
    echo "  - 使用 ./test_bandwidth_mpi_fixed.sh 重新测试"
else
    NUM_HOSTS=$(wc -l < hosts)
    if [ $NUM_HOSTS -eq 1 ]; then
        echo "⚠ hosts文件只有1个节点"
        echo "  无法测试多机带宽"
    else
        echo "✓ hosts文件有 $NUM_HOSTS 个节点"
        echo "  可以运行 ./test_bandwidth_mpi_fixed.sh 测试多机带宽"
    fi
fi

echo ""
echo "清理临时文件..."
rm -f test_mpi_distribution.cpp test_mpi_distribution
echo "完成！"

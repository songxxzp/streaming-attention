bash scripts/compile_attention.sh --clean

python3 compare_attention_full.py \
    --enable-mpi \
    --threads 1 2 4 8 16 26 32\
    --seqlen 1024 2048 4096 8192 16384 32768 65536 \
    --block-sizes 64 128 256 \
    --seqlen-scale 1024 8192 65536 \
    --block-size-mpi 128 \
    --mpi-ranks 1 2 4 8 16 26 32\
    --mpi-omp-threads 1 2 4 8 16 32\
    2>&1 | tee results.log

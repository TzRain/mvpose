#!/bin/bash

while true; do
    CUDA_VISIBLE_DEVICES=0 python src/tools/preprocess.py -d Campus -dump_dir ./datasets/Campus_processed
    sleep 1  # 等待一秒后重新运行命令，可以根据需要调整等待时间
done

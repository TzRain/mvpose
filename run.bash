#!/bin/bash

while true; do
    CUDA_VISIBLE_DEVICES=0 python ./src/m_utils/evaluate.py -d Campus
    sleep 1  # 等待一秒后重新运行命令，可以根据需要调整等待时间
done

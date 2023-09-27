#!/bin/bash
cam="$1"
seq_list=('160906_pizza1' '160422_haggling1' '160906_ian5' '160906_band4')
# 使用for循环迭代数组中的元素
for seq in "${seq_list[@]}"; do
    echo "Processing sequence: $seq"
    python ./src/m_utils/evaluate.py -d panoptic --seq $seq --cam $cam
done
    
    
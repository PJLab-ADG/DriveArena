#!/bin/bash

cuda_device=(0 1 2 3 4 5 6 7)
numbers=(0 755 1506 2266 3032 3759 4526 5296 6019)

set -x
for ((i = 0; i < ${#numbers[@]} - 1; i++))
do
    sta=${numbers[i]}
    end=${numbers[i+1]}
    CUDA_VISIBLE_DEVICES=${cuda_device[i]} python demo/run_for_eval.py start=$sta end=$end &
done

wait
#!/bin/bash
nohup bash -c '
for n in $(seq 2 10); do
    labelmask=$(seq 0 $((n-1)) | tr "\n" " ")
    python3 overhead-tree.py --dataset CIFAR --mode tree \
        --arch classical --labelmask $labelmask
done' > output.log 2>&1 &

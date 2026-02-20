#!/bin/bash
nohup bash -c '
for n in $(seq 2 10); do
    labelmask=$(seq 0 $((n-1)) | tr "\n" " ")
    python3 main.py --dataset MNIST --mode tree \
        --arch quantum --labelmask $labelmask
done' > output.log 2>&1 &

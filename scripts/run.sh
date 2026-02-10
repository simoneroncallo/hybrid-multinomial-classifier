#!/bin/bash
nohup bash -c '
for a in classical quantum; do
    for m in tree one_vs_rest one_vs_all; do
        for d in MNIST Fashion CIFAR; do
            python3 main.py --dataset "$d" --mode "$m" --arch "$a"
        done
    done
done' > output.log 2>&1 &

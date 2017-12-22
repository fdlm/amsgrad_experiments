#!/usr/bin/env bash

DEVICE=cuda0

for RUN in 0 1 2 3 4; do
    for LR in 0.001 0.01 0.0001; do
        for B2 in 0.99 0.999; do
            for UPD in amsgrad adam; do
                echo "========= RUN ${RUN}, ${UPD}, lr = ${LR}, beta2 = ${B2}"
                THEANO_FLAGS=device=${DEVICE} python train.py --model cifarnet --data cifar10 --n_epochs 150 \
                       --updater ${UPD} --learning_rate ${LR} --beta2 ${B2} --run_id ${RUN}
                echo "========= RUN ${RUN}, ${UPD}, lr = ${LR}, beta2 = ${B2}, bias correction on"
                THEANO_FLAGS=device=${DEVICE} python train.py --model cifarnet --data cifar10 --n_epochs 150 \
                       --updater ${UPD} --learning_rate ${LR} --beta2 ${B2} --run_id ${RUN} --bias_correction
            done
        done
    done
done

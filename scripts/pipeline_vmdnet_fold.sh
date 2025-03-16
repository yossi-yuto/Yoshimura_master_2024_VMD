#!/bin/bash

DEVICE=$1
EXE_NAME=$2

if [ -z "$DEVICE" ]; then
    echo "Error: GPU device ID must be provided as the first argument."
    exit 1
fi

for i in {0..4}; do
    experiment_dir=${EXE_NAME}_fold_${i}
    echo "This is the experiment_dir: ${experiment_dir}"
    echo "Processing fold $i..."
    python train.py --fold $i --exp ${experiment_dir} --batchsize 6 --gpu $DEVICE --bestonly
    python infer.py --param_path experiment_results/${experiment_dir}/best_mae.pth --result_path experiment_results/${experiment_dir}
done

echo "All folds processed successfully!"

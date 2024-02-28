#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset="cifar10"
class_names=(airplanes cars birds cats deer dogs frogs horses ships trucks)

for class_name in "${class_names[@]}"; do
    output_dir="/home/mila/d/diganta.misra/scratch/sd_xl_images/$dataset/$class_name"
    mkdir -p "$output_dir"

    sbatch /home/mila/d/diganta.misra/projects/Gen_data/mila.sh $class_name $output_dir

done

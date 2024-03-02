#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset="office_homes"
#class_names=("alarm clock" "backpack" "batteries" "bed" "bike" "bottle" "bucket" "calculator" "calendar" "candles" "chair" "clipboards" "computer" "couch" "curtains" "desk lamp" "drill" "eraser" "exit sign" "fan" "file cabinet" "flipflops" "flowers" "folder" "fork" "glasses" "hammer" "helmet" "kettle" "keyboard" "knives" "lamp shade" "laptop" "marker" "monitor" "mop" "mouse" "mug" "notebook" "oven" "pan" "paper clip" "pen" "pencil" "postit notes" "printer" "push pin" "radio" "refrigerator" "ruler" "scissors" "screwdriver" "shelf" "sink" "sneakers" "soda" "speaker" "spoon" "table" "telephone" "toothbrush" "toys" "trash can" "tv" "webcam")
class_names=("alarm clock" "desk lamp" "exit sign" "file cabinet" "lamp shade" "paper clip" "postit notes" "push pin" "trash can")
img_per_class=11

for class_name in "${class_names[@]}"; do
    output_dir="/home/mila/d/diganta.misra/scratch/floyd/$dataset/$class_name"
    mkdir -p "$output_dir"

    sbatch /home/mila/d/diganta.misra/projects/Gen_data/mila.sh $class_name $output_dir $img_per_class

done

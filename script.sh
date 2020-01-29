#!/usr/bin/env bash
python3 prepare_data.py
python3 monodepth_main.py --data_dir /data/data/kitti_training --val_data_dir /data/data/kitti_testing --model_path ./pretrained/monodepth_resnet18_002.pth --output_directory ./output --mode train
python3 find_best_parameters.py --root ./selected_data --id 11377
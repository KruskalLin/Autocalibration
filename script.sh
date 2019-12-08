#!/usr/bin/env bash
python3 prepare_data.py
python3 monodepth_main.py --data_dir /data/data/2011_09_26/whole_data --model_path ./pretrained/monodepth_resnet18_001.pth --output_directory ./selected_data --pretrained True --mode test
python3 find_best_parameters.py --root ./selected_data --id 11377
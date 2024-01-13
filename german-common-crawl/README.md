# USABLE GERMAN COMMON CRAWL DATASET

## less commands way
1. chmod +x preprocess.sh
2. ./preprocess.sh
3. python3 train.py

## How to create this dataset 
0. chmod +x convert_file.sh
1. download and unzip files: run `download_files.py`
2. move files to directory base_files
3. write files in correct json format: run `write_correct_files.py` 
4. `get_translation_pairs.py` 
5. `train.py --model_name <MODELNAME> --pretrained_model <HUGGINGFACE_CHECKPOINT>` only tested on 't5-small'
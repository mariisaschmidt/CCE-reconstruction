# USABLE GERMAN COMMON CRAWL DATASET & FINETUNING WITH ELLIPSIS DATA

## less commands way
1. chmod +x preprocess.sh
2. ./preprocess.sh
3. `finetune.py --dataset gcc --model_name <MODELNAME> --pretrained_model <HUGGINGFACE_CHECKPOINT>` only tested on 't5-small'
4. `finetune.py --dataset <tiger/tüba> --model_name <MODELNAME> --pretrained_model <PATH/TO/PRETRAINED/LLM>` 

### How to create this dataset 
0. chmod +x convert_file.sh
1. download and unzip files: run `download_files.py`
2. move files to directory base_files
3. write files in correct json format: run `write_correct_files.py` 
4. `get_translation_pairs.py` 
5. get_ellipsis_data.py

### how to train the model and finetune it for ellipsis
1. `finetune.py --dataset gcc --model_name <MODELNAME> --pretrained_model <HUGGINGFACE_CHECKPOINT>` only tested on 't5-small'
2. `finetune.py --dataset <tiger/tüba> --model_name <MODELNAME> --pretrained_model <PATH/TO/PRETRAINED/LLM>` 
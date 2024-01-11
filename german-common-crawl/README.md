# USABLE GERMAN COMMON CRAWL DATASET

## How to create this dataset 
0. chmod +x convert_file.sh
1. download and unzip files: run `download_files.py`
2. move files to directory base_files
3. write files in correct json format: run `write_correct_files.py` 
4. `get_translation_pairs.py` 
5. `train model with train.py`
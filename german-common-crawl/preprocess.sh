chmod +x convert_file.sh

python3 download_files.py

mkdir base_files
mv *.jsonl base_files/

mkdir correct_files

python3 write_correct_files.py

python3 get_translation_pairs.py
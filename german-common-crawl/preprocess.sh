chmod +x convert_file.sh

python3 download_files.py

mkdir base_files
mv *.jsonl base_files/

mkdir correct_files

python3 write_correct_files.py

rm -r base_files

python3 get_translation_pairs.py

rm -r correct_files

mkdir tmp_json

python3 get_ellipsis_data_tüba.py

rm -r tmp_json

mkdir tmp_json2

python3 get_ellipsis_data_tiger.py

rm -r tmp_json2
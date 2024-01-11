chmod +x convert_file.sh

python3 download_files.py

mkdir base_files
mv *.tar.gz basefiles/

python3 write_correct_files.py

python3 get_translation_pairs.py
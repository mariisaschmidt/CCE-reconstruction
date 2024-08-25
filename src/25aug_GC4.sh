# echo "Preparing GC4 corpus!"

# chmod +x convert_file.sh
# python3 download_files.py
# mkdir base_files
# mv *.jsonl base_files/
# mkdir correct_files
# python3 write_correct_files.py
# rm -r base_files
# python3 get_translation_pairs.py
# rm -r correct_files

echo "Start training of T5 Small"

python3 finetune.py --dataset g4 --model_name Aug25Small --pretrained_model t5-small

echo "Done!"
echo "Start training of T5 Base"

python3 finetune.py --dataset g4 --model_name Aug25Base --pretrained_model t5-base

echo "Done! :)"
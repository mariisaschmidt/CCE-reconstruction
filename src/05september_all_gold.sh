echo "Cleaning data: " # TODO: comment later
python3 clean_data.py 

# echo "Start training of TigerSmall"

# python3 finetune.py --dataset tiger --model_name  04allGoldsTigerSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

# echo "Start training of TübaSmall"

# python3 finetune.py --dataset tüba --model_name  04allGoldsTübaSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

# echo "Start training of TigerBase"

# python3 finetune.py --dataset tiger --model_name  04allGoldsTigerBase  --pretrained_model /home/marisa/models/de_de_mar14

# echo "Start training of TübaBase"

# python3 finetune.py --dataset tüba --model_name  04allGoldsTübaBase  --pretrained_model /home/marisa/models/de_de_mar14

# echo "Done with training!"

echo "Running evaluation:"

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTigerSmall  --corpus eval --prefix 05sep_TiSm

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTigerSmall  --corpus tuba --prefix 05sep_TiSm

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTigerSmall  --corpus tiger --prefix 05sep_TiSm

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTübaBase  --corpus eval --prefix 05sep_TuBas

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTübaBase  --corpus tuba --prefix 05sep_TuBas

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTübaBase  --corpus tiger --prefix 05sep_TuBas

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTigerBase  --corpus tuba --prefix 05sep_TiBas

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTigerBase  --corpus tiger --prefix 05sep_TiBas

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTigerBase  --corpus eval --prefix 05sep_TiBas

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTübaSmall  --corpus tuba --prefix 05sep_TuSm

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTübaSmall  --corpus tiger --prefix 05sep_TuSm

python3 test_model.py --checkpoint /home/marisa/models/04allGoldsTübaSmall  --corpus eval --prefix 05sep_TuSm

echo "Done with evaluation!"
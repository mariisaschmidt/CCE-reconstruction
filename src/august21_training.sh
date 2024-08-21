echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name cleanTigerSmall21 --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name cleanTübaSmall21 --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TigerBase"

python3 finetune.py --dataset tiger --model_name cleanTigerBase21 --pretrained_model /home/marisa/models/de_de_mar14

echo "Start training of TübaBase"

python3 finetune.py --dataset tüba --model_name cleanTübaBase21 --pretrained_model /home/marisa/models/de_de_mar14

echo "Done with training!"
echo "Running evaluation script:"

chmod +x august_eval.sh
./august_eval.sh 
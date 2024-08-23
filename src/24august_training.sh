echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  24cleanTigerSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name  24cleanTübaSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TigerBase"

python3 finetune.py --dataset tiger --model_name  24cleanTigerBase  --pretrained_model /home/marisa/models/de_de_mar14

echo "Start training of TübaBase"

python3 finetune.py --dataset tüba --model_name  24cleanTübaBase  --pretrained_model /home/marisa/models/de_de_mar14

echo "Done with training!"
echo "Running evaluation script:"

chmod +x 24august_eval.sh
./24august_eval.sh 
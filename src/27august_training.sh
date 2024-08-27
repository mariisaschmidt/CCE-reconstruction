echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  27cleanTigerSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name  27cleanTübaSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TigerBase"

python3 finetune.py --dataset tiger --model_name  27cleanTigerBase  --pretrained_model /home/marisa/models/de_de_mar14

echo "Start training of TübaBase"

python3 finetune.py --dataset tüba --model_name  27cleanTübaBase  --pretrained_model /home/marisa/models/de_de_mar14

echo "Done with training!"
echo "Running evaluation script:"

chmod +x 27august_eval.sh
./24august_eval.sh 
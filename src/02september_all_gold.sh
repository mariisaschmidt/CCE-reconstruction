echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  02cleanTigerSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name  02cleanTübaSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of TigerBase"

python3 finetune.py --dataset tiger --model_name  02cleanTigerBase  --pretrained_model /home/marisa/models/de_de_mar14

echo "Start training of TübaBase"

python3 finetune.py --dataset tüba --model_name  02cleanTübaBase  --pretrained_model /home/marisa/models/de_de_mar14

echo "Done with training!"
echo "Running evaluation:"

python3 test_model.py --checkpoint /home/marisa/models/02cleanTigerSmall  --corpus eval --prefix 02sep_TiSm

python3 test_model.py --checkpoint /home/marisa/models/02cleanTigerSmall  --corpus tuba --prefix 02sep_TiSm

python3 test_model.py --checkpoint /home/marisa/models/02cleanTigerSmall  --corpus tiger --prefix 02sep_TiSm

python3 test_model.py --checkpoint /home/marisa/models/02cleanTübaBase  --corpus eval --prefix 02sep_TuBas

python3 test_model.py --checkpoint /home/marisa/models/02cleanTübaBase  --corpus tuba --prefix 02sep_TuBas

python3 test_model.py --checkpoint /home/marisa/models/02cleanTübaBase  --corpus tiger --prefix 02sep_TuBas

python3 test_model.py --checkpoint /home/marisa/models/02cleanTigerBase  --corpus tuba --prefix 02sep_TiBas

python3 test_model.py --checkpoint /home/marisa/models/02cleanTigerBase  --corpus tiger --prefix 02sep_TiBas

python3 test_model.py --checkpoint /home/marisa/models/02cleanTigerBase  --corpus eval --prefix 02sep_TiBas

python3 test_model.py --checkpoint /home/marisa/models/02cleanTübaSmall  --corpus tuba --prefix 02sep_TuSm

python3 test_model.py --checkpoint /home/marisa/models/02cleanTübaSmall  --corpus tiger --prefix 02sep_TuSm

python3 test_model.py --checkpoint /home/marisa/models/02cleanTübaSmall  --corpus eval --prefix 02sep_TuSm

echo "Done with evaluation!"
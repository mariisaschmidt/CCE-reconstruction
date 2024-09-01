echo "Start training of MergedSmall"

python3 finetune.py --dataset merged --model_name  mergedSepSmall  --pretrained_model /home/marisa/models/de_de_feb05/checkpoint-929000

echo "Start training of MergedBase"

python3 finetune.py --dataset merged --model_name  mergedSepBase --pretrained_model /home/marisa/models/de_de_mar14

echo "Done with training!"
echo "Running evaluation:"

python3 test_model.py --checkpoint /home/marisa/models/mergedSepSmall --corpus tiger --prefix 01sep_merSm
python3 test_model.py --checkpoint /home/marisa/models/mergedSepSmall --corpus tuba --prefix 01sep_merSm
python3 test_model.py --checkpoint /home/marisa/models/mergedSepSmall --corpus eval --prefix 01sep_merSm

python3 test_model.py --checkpoint /home/marisa/models/mergedSepBase --corpus tiger --prefix 01sep_merBas
python3 test_model.py --checkpoint /home/marisa/models/mergedSepBase --corpus tuba --prefix 01sep_merBas
python3 test_model.py --checkpoint /home/marisa/models/mergedSepBase --corpus eval --prefix 01sep_merBas

echo "Done with evaluation!"
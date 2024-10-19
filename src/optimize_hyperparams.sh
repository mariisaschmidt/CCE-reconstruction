echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  19Okt_Optimized_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/19Okt_Optimized_OneNew_MergedBase --corpus tiger --prefix 19Okt_Opt

python3 test_model.py --checkpoint /home/marisa/models/19Okt_Optimized_OneNew_MergedBase --corpus tuba --prefix 19Okt_Opt

python3 test_model.py --checkpoint /home/marisa/models/19Okt_Optimized_OneNew_MergedBase --corpus eval --prefix 19Okt_Opt

echo "Done :)"
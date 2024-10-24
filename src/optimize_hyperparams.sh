echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  24Okt_EM_Optimized_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/24Okt_EM_Optimized_OneNew_MergedBase --corpus tiger --prefix 24Okt_EM_Opt

python3 test_model.py --checkpoint /home/marisa/models/24Okt_EM_Optimized_OneNew_MergedBase --corpus tuba --prefix 24Okt_EM_Opt

python3 test_model.py --checkpoint /home/marisa/models/24Okt_EM_Optimized_OneNew_MergedBase --corpus eval --prefix 24Okt_EM_Opt

echo "Done :)"
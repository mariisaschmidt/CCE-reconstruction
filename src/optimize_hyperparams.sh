# echo "Start optimization of hyperparams"

# python3 finetune.py --dataset merged --model_name  14Okt_Optimized_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/14Okt_Optimized_OneNew_MergedBase --corpus tiger --prefix 14Okt_Opt_

python3 test_model.py --checkpoint /home/marisa/models/14Okt_Optimized_OneNew_MergedBase --corpus tuba --prefix 14Okt_Opt_

python3 test_model.py --checkpoint /home/marisa/models/14Okt_Optimized_OneNew_MergedBase --corpus eval --prefix 14Okt_Opt_

echo "Done :)"
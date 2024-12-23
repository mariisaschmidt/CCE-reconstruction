echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  23Dec_BLEU_Optimized_AllNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/23Dec_BLEU_Optimized_AllNew_MergedBase --corpus tiger --prefix 23Dec_BLEU_Opt

python3 test_model.py --checkpoint /home/marisa/models/23Dec_BLEU_Optimized_AllNew_MergedBase --corpus tuba --prefix 23Dec_BLEU_Opt

python3 test_model.py --checkpoint /home/marisa/models/23Dec_BLEU_Optimized_AllNew_MergedBase --corpus eval --prefix 23Dec_BLEU_Opt

echo "Done :)"
echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  30Dec_BLEU_Optimized_AllNewFE_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/30Dec_BLEU_Optimized_AllNewFE_MergedBase --corpus tiger --prefix 30Dec_BLEU_Opt

python3 test_model.py --checkpoint /home/marisa/models/30Dec_BLEU_Optimized_AllNewFE_MergedBase --corpus tuba --prefix 30Dec_BLEU_Opt

python3 test_model.py --checkpoint /home/marisa/models/30Dec_BLEU_Optimized_AllNewFE_MergedBase --corpus eval --prefix 30Dec_BLEU_Opt

echo "Done :)"
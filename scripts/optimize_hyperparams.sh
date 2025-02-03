cd ../src

echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  01Jan_EM+BLEU_Optimized_AllNewFE_MergedBase  --pretrained_model ../models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint ../models/01Jan_EM+BLEU_Optimized_AllNewFE_MergedBase --corpus merged --prefix 01Jan_EM+BLEU_Opt

echo "Done :)"

cd ../
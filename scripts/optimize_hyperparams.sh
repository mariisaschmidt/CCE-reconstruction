cd ../src

echo "Start optimization of hyperparameters"

python3 finetune.py --dataset merged --model_name  EM+BLEU_Optimized_AllNewFE_MergedBase  --pretrained_model ../models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint ../models/01Jan_EM+BLEU_Optimized_AllNewFE_MergedBase --corpus merged --prefix EM+BLEU_Opt

echo "Done :)"

cd ../
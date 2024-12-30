echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  31Dec_EM_Optimized_AllNewFE_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/31Dec_EM_Optimized_AllNewFE_MergedBase --corpus merged --prefix 31Dec_EM_Opt

echo "Done :)"
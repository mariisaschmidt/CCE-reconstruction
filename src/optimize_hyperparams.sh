echo "Start optimization of hyperparams"

python3 finetune.py --dataset merged --model_name  25sep_Optimized_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Done :)"
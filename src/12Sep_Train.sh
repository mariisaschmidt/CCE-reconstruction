echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  12sep_AllOld_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name  12sep_AllOld_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small

echo "Start training of MergedSmall"

python3 finetune.py --dataset merged --model_name  12sep_AllOld_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small

# echo "Start training of TigerBase"

# python3 finetune.py --dataset tiger --model_name  12sep_OneOld_TigerBase  --pretrained_model /home/marisa/models/Aug25Base

# echo "Start training of TübaBase"

# python3 finetune.py --dataset tüba --model_name  12sep_OneOld_TübaBase  --pretrained_model /home/marisa/models/Aug25Base

# echo "Start training of MergedBase"

# python3 finetune.py --dataset merged --model_name  12sep_OneOld_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Done with training!"
echo "Running evaluation script:"

chmod +x 12sep_eval.sh
./12sep_eval.sh 
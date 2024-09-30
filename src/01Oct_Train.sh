echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  01oct_NoNoCCE_OneOld_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name  01oct_NoNoCCE_OneOld_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1

echo "Start training of MergedSmall"

python3 finetune.py --dataset merged --model_name  01oct_NoNoCCE_OneOld_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1

echo "Start training of TigerBase"

python3 finetune.py --dataset tiger --model_name  01oct_NoNoCCE_OneOld_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1

echo "Start training of TübaBase"

python3 finetune.py --dataset tüba --model_name  01oct_NoNoCCE_OneOld_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1

echo "Start training of MergedBase"

python3 finetune.py --dataset merged --model_name  01oct_NoNoCCE_OneOld_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1

echo "Done with training!"
echo "Running evaluation script:"

chmod +x 01oct_eval.sh
./01oct_eval.sh 
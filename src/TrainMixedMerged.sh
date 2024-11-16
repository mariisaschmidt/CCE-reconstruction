echo "Start training of MergedSmall"

python3 finetune.py --dataset merged --model_name  16nov_Mixed_AllNew_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small

echo "Start training of MergedBase"

python3 finetune.py --dataset merged --model_name  16nov_Mixed_AllNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Done with training!"

echo "Evaluating..."

python3 test_model.py --checkpoint /home/marisa/models/16nov_Mixed_AllNew_MergedSmall --corpus merged --prefix 1611_MixedAllNew_MergedSmall  

python3 test_model.py --checkpoint /home/marisa/models/16nov_Mixed_AllNew_MergedBase --corpus merged --prefix 1611_MixedAllNew_MergedBase

echo "Done with Evaluation"
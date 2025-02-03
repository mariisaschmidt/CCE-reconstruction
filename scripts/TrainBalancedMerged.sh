cd ../src/

echo "Start training of MergedSmall"

python3 finetune.py --dataset mergedFair --model_name  17nov_Fair_AllNew_MergedSmall  --pretrained_model ../models/Aug25Small

echo "Start training of MergedBase"

python3 finetune.py --dataset mergedFair --model_name  17nov_Fair_AllNew_MergedBase  --pretrained_model ../models/Aug25Base

echo "Done with training!"

echo "Evaluating..."

python3 test_model.py --checkpoint ../models/17nov_Fair_AllNew_MergedSmall --corpus merged --prefix 1711_FairAllNew_MergedSmall  

python3 test_model.py --checkpoint ../models/17nov_Fair_AllNew_MergedBase --corpus merged --prefix 1711_FairAllNew_MergedBase

echo "Done with Evaluation"

cd ../
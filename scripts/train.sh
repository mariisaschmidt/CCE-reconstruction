read -p "Please enter a name for your model: " modelname
read -p "Which dataset do you want to train on?" dataset
read -p "Which checkpoint should be used as reference? (Path)" checkpoint

cd ../src

python3 finetune.py --dataset dataset --model_name modelname --pretrained_model checkpoint

cd ..
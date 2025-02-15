echo "Train the models with the new variant of TIGER."

cd ../src

echo "Start training of: Tiger Erweitert"
echo "TigerSmall"
python3 finetune.py --dataset erwTiger --model_name 10Ep_ErwTigerSmall --pretrained_model ../models/Aug25Small --remove_no_cce 1

echo "TigerBase"
python3 finetune.py --dataset erwTiger --model_name 10Ep_ErwTigerBase --pretrained_model ../models/Aug25Base --remove_no_cce 1

echo "TigerSmall"
python3 finetune.py --dataset erwTiger --model_name 5Ep_ErwTigerSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0

echo "TigerBase"
python3 finetune.py --dataset erwTiger --model_name 5Ep_ErwTigerBase --pretrained_model ../models/Aug25Base --remove_no_cce 0

python3 test_model.py --checkpoint ../models/10Ep_ErwTigerSmall  --corpus merged --prefix 10Ep_ErwTiSm

python3 test_model.py --checkpoint ../models/10Ep_ErwTigerBase  --corpus merged --prefix 10Ep_ErwTiBas

python3 test_model.py --checkpoint ../models/5Ep_ErwTigerSmall  --corpus merged --prefix 5Ep_ErwTiSm

python3 test_model.py --checkpoint ../models/5Ep_ErwTigerBase  --corpus merged --prefix 5Ep_ErwTiBas

cd ../
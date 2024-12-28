echo "Start training of: Tiger Erweitert"
echo "Start training of TigerSmall"
python3 finetune.py --dataset erwTiger --model_name 2812_10Ep_ErwTigerSmall --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1

echo "Start training of TigerBase"
python3 finetune.py --dataset erwTiger --model_name 2812_10Ep_ErwTigerBase --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1

echo "Start training of TigerSmall"
python3 finetune.py --dataset erwTiger --model_name 2812_5Ep_ErwTigerSmall --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0

echo "Start training of TigerBase"
python3 finetune.py --dataset erwTiger --model_name 2812_5Ep_ErwTigerBase --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0

python3 test_model.py --checkpoint /home/marisa/models/2812_10Ep_ErwTigerSmall  --corpus merged --prefix 2812_10Ep_ErwTiSm

python3 test_model.py --checkpoint /home/marisa/models/2812_10Ep_ErwTigerBase  --corpus merged --prefix 2812_10Ep_ErwTiBas

python3 test_model.py --checkpoint /home/marisa/models/2812_5Ep_ErwTigerSmall  --corpus merged --prefix 2812_5Ep_ErwTiSm

python3 test_model.py --checkpoint /home/marisa/models/2812_5Ep_ErwTigerBase  --corpus merged --prefix 2812_5Ep_ErwTiBas
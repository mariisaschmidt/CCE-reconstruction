echo "Start training of: Tiger Erweitert"
echo "Start training of TigerSmall"
python3 finetune.py --dataset erwTiger --model_name 2812_ErwTigerSmall --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0

echo "Start training of TigerBase"
python3 finetune.py --dataset erwTiger --model_name 2812_ErwTigerBase --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0

python3 test_model.py --checkpoint /home/marisa/models/2812_ErwTigerSmall  --corpus merged --prefix 2812_ErwTiSm

python3 test_model.py --checkpoint /home/marisa/models/2812_ErwTigerBase  --corpus merged --prefix 2812_ErwTiBas
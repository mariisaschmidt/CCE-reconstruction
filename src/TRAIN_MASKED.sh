echo "Start training of CCEFairMaskedSmall"
python3 finetuneMasking.py --model_name  1101_5Ep_CCEFairMaskedSmall --pretrained_model /home/marisa/models/Aug25Small

echo "Start evaluation of CCEFairMaskedSmall"
python3 testMasking.py --checkpoint /home/marisa/models/1101_5Ep_CCEFairMaskedSmall --prefix 1101_5Ep_maskedSmall

echo "Start training of CCEFairMaskedBase"
python3 finetuneMasking.py --model_name  1101_5Ep_CCEFairMaskedBase --pretrained_model /home/marisa/models/Aug25Base

echo "Start evaluation of CCEFairMaskedBase"
python3 testMasking.py --checkpoint /home/marisa/models/1101_5Ep_CCEFairMaskedBase --prefix 1101_5Ep_maskedBase
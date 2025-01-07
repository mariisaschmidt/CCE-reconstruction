echo "Start training of FairMaskedSmall"
python3 finetuneMasking.py --model_name  0601_10Ep_FairMaskedSmall --pretrained_model /home/marisa/models/Aug25Small

echo "Start evaluation of FairMaskedSmall"
python3 testMasking.py --checkpoint /home/marisa/models/0601_10Ep_FairMaskedSmall --prefix 0601_10Ep_maskedSmall

echo "Start training of FairMaskedBase"
python3 finetuneMasking.py --model_name  0601_10Ep_FairMaskedBase --pretrained_model /home/marisa/models/Aug25Base

echo "Start evaluation of FairMaskedBase"
python3 testMasking.py --checkpoint /home/marisa/models/0601_10Ep_FairMaskedBase --prefix 0601_10Ep_maskedBase
echo "Start training of MaskedSmall"
python3 finetuneMasking.py --model_name  0212_MaskedSmall --pretrained_model /home/marisa/models/Aug25Small

echo "Start evaluation of MaskedSmall"
python3 testMasking.py --checkpoint /home/marisa/models/0212MaskedSmall --prefix 0212_maskedSmall

echo "Start training of MaskedBase"
python3 finetuneMasking.py --model_name  0212_MaskedBase --pretrained_model /home/marisa/models/Aug25Base

echo "Start evaluation of MaskedBase"
python3 testMasking.py --checkpoint /home/marisa/models/0212MaskedBase --prefix 0212_maskedBase
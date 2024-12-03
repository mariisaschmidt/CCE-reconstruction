# echo "Start training of MaskedSmall"
# python3 finetuneMasking.py --model_name  0312_MaskedSmall --pretrained_model /home/marisa/models/Aug25Small

echo "Start evaluation of MaskedSmall"
python3 testMasking.py --checkpoint /home/marisa/models/0312_MaskedSmall --prefix 0312_maskedSmall

# echo "Start training of MaskedBase"
# python3 finetuneMasking.py --model_name  0312_MaskedBase --pretrained_model /home/marisa/models/Aug25Base

echo "Start evaluation of MaskedBase"
python3 testMasking.py --checkpoint /home/marisa/models/0312_MaskedBase --prefix 0312_maskedBase
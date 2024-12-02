python3 finetuneMasking.py --model_name  0212_MaskedSmall --pretrained_model /home/marisa/models/Aug25Small

python3 testMasking.py --checkpoint /home/marisa/models/0212MaskedSmall --prefix 0212_maskedSmall

python3 finetuneMasking.py --model_name  0212_MaskedBase --pretrained_model /home/marisa/models/Aug25Base

python3 testMasking.py --checkpoint /home/marisa/models/0212MaskedBase --prefix 0212_maskedBase
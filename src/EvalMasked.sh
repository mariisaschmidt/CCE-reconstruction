echo "Start evaluation of Masked Small"
python3 test_model.py --checkpoint /home/marisa/models/0601_10Ep_maskedSmall --corpus merged --prefix 0601_NoMaskedEval_MaskedSmall

python3 test_model.py --checkpoint /home/marisa/models/0601_5Ep_maskedSmall --corpus merged --prefix 0601_NoMaskedEval_5Ep_MaskedSmall

echo "Start evaluation of Masked Base"
python3 test_model.py --checkpoint /home/marisa/models/0601_10Ep_maskedBase --corpus merged --prefix 0601_NoMaskedEval_MaskedBase

python3 test_model.py --checkpoint /home/marisa/models/0601_5Ep_maskedBase --corpus merged --prefix 0601_NoMaskedEval_5Ep_MaskedBase
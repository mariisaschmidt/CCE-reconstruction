echo "Start evaluation of Masked Small"
python3 test_model.py --checkpoint /home/marisa/models/0412_MaskedSmall --corpus merged --prefix 0512_NoMaskedEval_MaskedSmall

python3 test_model.py --checkpoint /home/marisa/models/0512_5Ep_MaskedSmall --corpus merged --prefix 0512_NoMaskedEval_5Ep_MaskedSmall

echo "Start evaluation of Masked Base"
python3 test_model.py --checkpoint /home/marisa/models/0412_MaskedBase --corpus merged --prefix 0512_NoMaskedEval_MaskedBase

python3 test_model.py --checkpoint /home/marisa/models/0512_5Ep_MaskedBase --corpus merged --prefix 0512_NoMaskedEval_5Ep_MaskedBase
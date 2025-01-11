echo "Start evaluation of Masked Small"
python3 test_model.py --checkpoint /home/marisa/models/0601_10Ep_FairMaskedSmall --corpus merged --prefix GettingDataForDenis

# python3 test_model.py --checkpoint /home/marisa/models/0601_5Ep_FairMaskedSmall --corpus merged --prefix 0601_NoMaskedEval_5Ep_MaskedSmall

# echo "Start evaluation of Masked Base"
# python3 test_model.py --checkpoint /home/marisa/models/0601_10Ep_FairMaskedBase --corpus merged --prefix 0601_NoMaskedEval_MaskedBase

# python3 test_model.py --checkpoint /home/marisa/models/0601_5Ep_FairMaskedBase --corpus merged --prefix 0601_NoMaskedEval_5Ep_MaskedBase
echo "Start evaluation of Masked 5 Eps"
python3 test_model.py --checkpoint /home/marisa/models/1101_5Ep_CCEFairMaskedSmall --corpus merged --prefix 1201_5Ep_NoMasked_Masked+CCE_Small

python3 test_model.py --checkpoint /home/marisa/models/1101_5Ep_CCEFairMaskedBase --corpus merged --prefix 1201_5Ep_NoMasked_Masked+CCE_Base

echo "Start evaluation of Masked 10 Eps"
python3 test_model.py --checkpoint /home/marisa/models/1101_10Ep_CCEFairMaskedSmall --corpus merged --prefix 1201_10Ep_NoMasked_Masked+CCE_Small

python3 test_model.py --checkpoint /home/marisa/models/1101_10Ep_CCEFairMaskedBase --corpus merged --prefix 1201_10Ep_NoMasked_Masked+CCE_Base
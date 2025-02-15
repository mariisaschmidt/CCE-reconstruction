echo "Evaluating the masked models on unmasked data."

cd ../src

echo "5 epochs models"
python3 test_model.py --checkpoint ../models/1101_5Ep_CCEFairMaskedSmall --corpus merged --prefix 5Ep_NoMasked_Masked+CCE_Small

python3 test_model.py --checkpoint ../models/1101_5Ep_CCEFairMaskedBase --corpus merged --prefix 5Ep_NoMasked_Masked+CCE_Base

echo "10 epochs models"
python3 test_model.py --checkpoint ../models/1101_10Ep_CCEFairMaskedSmall --corpus merged --prefix 10Ep_NoMasked_Masked+CCE_Small

python3 test_model.py --checkpoint ../models/1101_10Ep_CCEFairMaskedBase --corpus merged --prefix 10Ep_NoMasked_Masked+CCE_Base

cd ../
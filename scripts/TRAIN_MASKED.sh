echo "Train models on masked data based on AG+AD_FE."

cd ../src

echo "Start training of CCEFairMaskedSmall"
python3 finetune_masked.py --model_name  5Ep_CCEFairMaskedSmall --pretrained_model ../models/Aug25Small

echo "Start evaluation of CCEFairMaskedSmall"
python3 test_model_masked.py --checkpoint ../models/5Ep_CCEFairMaskedSmall --prefix 5Ep_maskedSmall_Masked-CCE

python3 test_model_masked.py --checkpoint ../models/10Ep_CCEFairMaskedSmall --prefix 10Ep_maskedSmall_Masked-CCE

echo "Start training of CCEFairMaskedBase"
python3 finetune_masked.py --model_name  5Ep_CCEFairMaskedBase --pretrained_model ../models/Aug25Base

echo "Start evaluation of CCEFairMaskedBase"
python3 test_model_masked.py --checkpoint ../models/5Ep_CCEFairMaskedBase --prefix 5Ep_maskedBase_Masked-CCE

python3 test_model_masked.py --checkpoint ../models/10Ep_CCEFairMaskedBase --prefix 10Ep_maskedBase_Masked-CCE

cd ../
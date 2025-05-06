echo "Train the models with the new variant of TIGER combined with TüBa."

# cd ../src

# echo "Start training of: TigerErw+TüBa Mixed One New Mixed"

# echo "TigerBase"
# python3 finetune.py --dataset ECBAE --model_name 10Ep_TErw+TB_Base --pretrained_model ../models/Aug25Base --remove_no_cce 0

# python3 test_model.py --checkpoint ../models/10Ep_TErw+TB_Base --corpus merged --prefix 05052025_ecbae_TErw+TB_Base

# cd ../

echo "Train models on masked data of the new variant of TIGER combined with TüBa"

cd ../src

echo "Start training of MaskedBase"
python3 finetune_masked.py --model_name  10Ep_TErw+TB_Base_Masked --pretrained_model ../models/Aug25Base

python3 test_model_masked.py --checkpoint ../models/10Ep_TErw+TB_Base_Masked --prefix 06052025_ECBAE_Masked_MaskedData

python3 test_model.py --checkpoint ../models/10Ep_TErw+TB_Base_Masked --corpus merged --prefix 06052025_ECBAE_Masked-NoMaskedData

cd ../
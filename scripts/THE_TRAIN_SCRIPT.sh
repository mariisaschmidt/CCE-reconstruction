echo "The all in one train script."

cd ../src

echo "Start training of: TSD"
echo "TigerSmall"
python3 finetune.py --dataset tiger --model_name  TSD_OneOld_TigerSmall  --pretrained_model ../models/de_de_feb05/checkpoint-929000 --remove_no_cce 0 --data_variant TSD

echo "TübaSmall"
python3 finetune.py --dataset tüba --model_name  TSD_OneOld_TübaSmall  --pretrained_model ../models/de_de_feb05/checkpoint-929000 --remove_no_cce 0 --data_variant TSD

echo "MergedSmall"
python3 finetune.py --dataset merged --model_name  TSD_OneOld_MergedSmall  --pretrained_model ../models/de_de_feb05/checkpoint-929000 --remove_no_cce 0 --data_variant TSD

# echo "TigerBase"
# python3 finetune.py --dataset tiger --model_name  TSD_OneOld_TigerBase  --pretrained_model ../models/de_de_mar14 --remove_no_cce 0 --data_variant TSD

# echo "TübaBase"
# python3 finetune.py --dataset tüba --model_name  TSD_OneOld_TübaBase  --pretrained_model ../models/de_de_mar14 --remove_no_cce 0 --data_variant TSD

# echo "MergedBase"
# python3 finetune.py --dataset merged --model_name  TSD_OneOld_MergedBase  --pretrained_model ../models/de_de_mar14 --remove_no_cce 0 --data_variant TSD

echo "============================================================="

echo "Start training of: Exact Match / One Old"
echo "TigerSmall"
python3 finetune.py --dataset tiger --model_name 10Ep_OneOld_TigerSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant OneOld

echo "TübaSmall"
python3 finetune.py --dataset tüba --model_name 10Ep_OneOld_TübaSmall  --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant OneOld

echo "MergedSmall"
python3 finetune.py --dataset merged --model_name  10Ep_OneOld_MergedSmall   --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant OneOld

# echo "TigerBase"
# python3 finetune.py --dataset tiger --model_name  10Ep_OneOld_TigerBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant OneOld

# echo "TübaBase"
# python3 finetune.py --dataset tüba --model_name  10Ep_OneOld_TübaBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant OneOld

# echo "MergedBase"
# python3 finetune.py --dataset merged --model_name  10Ep_OneOld_MergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant OneOld

echo "============================================================="
echo "Start training of: One Old No No CCE"
echo "TigerSmall"
python3 finetune.py --dataset tiger --model_name 10Ep_NoNoCCE_OneOld_TigerSmall   --pretrained_model ../models/Aug25Small --remove_no_cce 1 --data_variant OneOld

echo "TübaSmall"
python3 finetune.py --dataset tüba --model_name 10Ep_NoNoCCE_OneOld_TübaSmall  --pretrained_model ../models/Aug25Small --remove_no_cce 1 --data_variant OneOld

echo "MergedSmall"
python3 finetune.py --dataset merged --model_name 10Ep_NoNoCCE_OneOld_MergedSmall --pretrained_model ../models/Aug25Small --remove_no_cce 1 --data_variant OneOld

# echo "TigerBase"
# python3 finetune.py --dataset tiger --model_name  10Ep_NoNoCCE_OneOld_TigerBase  --pretrained_model ../models/Aug25Base --remove_no_cce 1 --data_variant OneOld

# echo "TübaBase"
# python3 finetune.py --dataset tüba --model_name  10Ep_NoNoCCE_OneOld_TübaBase  --pretrained_model ../models/Aug25Base --remove_no_cce 1 --data_variant OneOld

# echo "MergedBase"
# python3 finetune.py --dataset merged --model_name  10Ep_NoNoCCE_OneOld_MergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 1 --data_variant OneOld

echo "============================================================="
echo "Start training of: One New"
echo "TigerSmall"
python3 finetune.py --dataset tiger --model_name 10Ep_OneNew_TigerSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant OneNew

echo "TübaSmall"
python3 finetune.py --dataset tüba --model_name 10Ep_OneNew_TübaSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant OneNew

echo "MergedSmall"
python3 finetune.py --dataset merged --model_name 10Ep_OneNew_MergedSmall  --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant OneNew

# echo "TigerBase"
# python3 finetune.py --dataset tiger --model_name  10Ep_OneNew_TigerBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant OneNew

# echo "TübaBase"
# python3 finetune.py --dataset tüba --model_name  10Ep_OneNew_TübaBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant OneNew

# echo "MergedBase"
# python3 finetune.py --dataset merged --model_name  10Ep_OneNew_MergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant OneNew

echo "============================================================="
echo "Start training of: All Old"
echo "TigerSmall"
python3 finetune.py --dataset tiger --model_name 10Ep_AllOld_TigerSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllOld

echo "TübaSmall"
python3 finetune.py --dataset tüba --model_name 10Ep_AllOld_TübaSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllOld

echo "MergedSmall"
python3 finetune.py --dataset merged --model_name 10Ep_AllOld_MergedSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllOld

# echo "TigerBase"
# python3 finetune.py --dataset tiger --model_name  10Ep_AllOld_TigerBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllOld

# echo "TübaBase"
# python3 finetune.py --dataset tüba --model_name  10Ep_AllOld_TübaBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllOld

# echo "MergedBase"
# python3 finetune.py --dataset merged --model_name  10Ep_AllOld_MergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllOld

echo "============================================================="
echo "Start training of: All New"
echo "TigerSmall"
python3 finetune.py --dataset tiger --model_name 10Ep_AllNew_TigerSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "TübaSmall"
python3 finetune.py --dataset tüba --model_name 10Ep_AllNew_TübaSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "MergedSmall"
python3 finetune.py --dataset merged --model_name 10Ep_AllNew_MergedSmall --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllNew

# echo "TigerBase"
# python3 finetune.py --dataset tiger --model_name  10Ep_AllNew_TigerBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllNew

# echo "TübaBase"
# python3 finetune.py --dataset tüba --model_name  10Ep_AllNew_TübaBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllNew

# echo "MergedBase"
# python3 finetune.py --dataset merged --model_name  10Ep_AllNew_MergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New Mixed"

echo "MergedSmall"
python3 finetune.py --dataset mergedMixed --model_name  10Ep_AllNew_MixedMergedSmall  --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllNew

# echo "MergedBase"
# python3 finetune.py --dataset mergedMixed --model_name  10Ep_AllNew_MixedMergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New 5050"
python3 finetune.py --dataset mergedFair --model_name 10Ep_AllNew_FairMergedSmall  --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllNew

# echo "MergedBase"
# python3 finetune.py --dataset mergedFair --model_name  10Ep_AllNew_FairMergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New 5050 Large"
python3 finetune.py --dataset mergedFairLarge --model_name 5Eps_AllNew_LaFairMergedBase  --pretrained_model ../models/Aug25Small --remove_no_cce 0 --data_variant AllNew

# echo "MergedBase"
# python3 finetune.py --dataset mergedFairLarge --model_name  5Eps_AllNew_LaFairMergedBase  --pretrained_model ../models/Aug25Base --remove_no_cce 0 --data_variant AllNew

# echo "====================== EVAL ================================="
# echo "============================================================="
# echo "Start evaluation of: TSD"
# python3 test_model.py --checkpoint ../models/TSD_OneOld_TigerSmall  --corpus tiger --prefix TSD_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TübaSmall  --corpus tiger --prefix TSD_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_MergedSmall  --corpus tiger --prefix TSD_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TigerBase  --corpus tiger --prefix TSD_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TübaBase  --corpus tiger --prefix TSD_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_MergedBase  --corpus tiger --prefix TSD_OneOld_MeBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TigerSmall  --corpus tuba --prefix TSD_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TübaSmall  --corpus tuba --prefix TSD_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_MergedSmall  --corpus tuba --prefix TSD_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TigerBase  --corpus tuba --prefix TSD_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TübaBase  --corpus tuba --prefix TSD_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_MergedBase  --corpus tuba --prefix TSD_OneOld_MeBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TigerSmall  --corpus eval --prefix TSD_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TübaSmall  --corpus eval --prefix TSD_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_MergedSmall  --corpus eval --prefix TSD_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TigerBase  --corpus eval --prefix TSD_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_TübaBase  --corpus eval --prefix TSD_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/TSD_OneOld_MergedBase  --corpus eval --prefix TSD_OneOld_MeBas

# echo "============================================================="

# echo "Start evaluation of: Exact Match / One Old"
# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerSmall  --corpus eval --prefix 10Ep_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerSmall  --corpus tuba --prefix 10Ep_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerSmall  --corpus tiger --prefix 10Ep_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerSmall  --corpus merged --prefix 10Ep_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaSmall  --corpus tuba --prefix 10Ep_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaSmall  --corpus tiger --prefix 10Ep_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaSmall  --corpus eval --prefix 10Ep_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaSmall  --corpus merged --prefix 10Ep_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedSmall  --corpus eval --prefix 10Ep_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedSmall  --corpus tuba --prefix 10Ep_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedSmall  --corpus tiger --prefix 10Ep_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedSmall  --corpus merged --prefix 10Ep_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaBase  --corpus eval --prefix 10Ep_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaBase  --corpus tuba --prefix 10Ep_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaBase  --corpus tiger --prefix 10Ep_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TübaBase  --corpus merged --prefix 10Ep_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerBase  --corpus tuba --prefix 10Ep_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerBase  --corpus tiger --prefix 10Ep_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerBase  --corpus eval --prefix 10Ep_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_TigerBase  --corpus merged --prefix 10Ep_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedBase  --corpus eval --prefix 10Ep_OneOld_MeBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedBase  --corpus tuba --prefix 10Ep_OneOld_MeBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedBase  --corpus tiger --prefix 10Ep_OneOld_MeBas

# python3 test_model.py --checkpoint ../models/10Ep_OneOld_MergedBase  --corpus merged --prefix 10Ep_OneOld_MeBas

# echo "============================================================="
# echo "Start evaluation of: One Old No No CCE"
# python3 test_model.py --checkpoint ../models/10Ep_NoNoCCE_OneOld_TigerSmall  --corpus merged --prefix 10Ep_NoNoCCE_OneOld_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_NoNoCCE_OneOld_TübaSmall  --corpus merged --prefix 10Ep_NoNoCCE_OneOld_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_NoNoCCE_OneOld_MergedSmall  --corpus merged --prefix 10Ep_NoNoCCE_OneOld_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_NoNoCCE_OneOld_TigerBase  --corpus merged --prefix 10Ep_NoNoCCE_OneOld_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_NoNoCCE_OneOld_TübaBase  --corpus merged --prefix 10Ep_NoNoCCE_OneOld_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_NoNoCCE_OneOld_MergedBase  --corpus merged --prefix 10Ep_NoNoCCE_OneOld_MeBas

# echo "============================================================="
# echo "Start evaluation of: One New"
# python3 test_model.py --checkpoint ../models/10Ep_OneNew_TigerSmall  --corpus merged --prefix 10Ep_OneNew_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_OneNew_TübaSmall  --corpus merged --prefix 10Ep_OneNew_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_OneNew_MergedSmall  --corpus merged --prefix 10Ep_OneNew_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_OneNew_TigerBase  --corpus merged --prefix 10Ep_OneNew_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_OneNew_TübaBase  --corpus merged --prefix 10Ep_OneNew_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_OneNew_MergedBase  --corpus merged --prefix 10Ep_OneNew_MeBas

# echo "============================================================="
# echo "Start evaluation of: All Old"
# python3 test_model.py --checkpoint ../models/10Ep_AllOld_TigerSmall  --corpus merged --prefix 10Ep_AllOld_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_AllOld_TübaSmall  --corpus merged --prefix 10Ep_AllOld_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_AllOld_MergedSmall  --corpus merged --prefix 10Ep_AllOld_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_AllOld_TigerBase  --corpus merged --prefix 10Ep_AllOld_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_AllOld_TübaBase  --corpus merged --prefix 10Ep_AllOld_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_AllOld_MergedBase  --corpus merged --prefix 10Ep_AllOld_MeBas

# echo "============================================================="
# echo "Start evaluation of: All New"
# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TigerSmall  --corpus merged --prefix 10Ep_AllNew_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TübaSmall  --corpus merged --prefix 10Ep_AllNew_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_MergedSmall  --corpus merged --prefix 10Ep_AllNew_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TigerBase  --corpus merged --prefix 10Ep_AllNew_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TübaBase  --corpus merged --prefix 10Ep_AllNew_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_MergedBase  --corpus merged --prefix 10Ep_AllNew_MeBas

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TigerSmall  --corpus merged50 --prefix 10Ep_AllNew50_TiSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TübaSmall  --corpus merged50 --prefix 10Ep_AllNew50_TuSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_MergedSmall  --corpus merged50 --prefix 10Ep_AllNew50_MeSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TigerBase  --corpus merged50 --prefix 10Ep_AllNew50_TiBas

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_TübaBase  --corpus merged50 --prefix 10Ep_AllNew50_TuBas

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_MergedBase  --corpus merged50 --prefix 10Ep_AllNew50_MeBas

# echo "============================================================="
# echo "Start evaluation of: All New Mixed"
# python3 test_model.py --checkpoint ../models/10Ep_AllNew_MixedMergedSmall  --corpus merged --prefix 10Ep_AllNew_MixMeSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_MixedMergedBase  --corpus merged --prefix 10Ep_AllNew_MixMeBas

# echo "============================================================="
# echo "Start evaluation of: All New 5050"
# python3 test_model.py --checkpoint ../models/10Ep_AllNew_FairMergedSmall  --corpus merged --prefix 10Ep_AllNew_FaiMeSm

# python3 test_model.py --checkpoint ../models/10Ep_AllNew_FairMergedBase  --corpus merged --prefix 10Ep_AllNew_FaiMeBas

# echo "============================================================="
# echo "Start evaluation of: All New 5050 Large"
# python3 test_model.py --checkpoint ../models/10Eps_AllNew_LaFairMergedSmall --corpus merged --prefix 10Ep_AllNew_LaFaiMeSm

# python3 test_model.py --checkpoint ../models/AllNew_LaFairMergedBase  --corpus merged --prefix 10Ep_AllNew_LaFaiMeBas

# echo "DONE DONE DONE!"

# cd ../
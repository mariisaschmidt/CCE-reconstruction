echo "Start training of: Exact Match / One Old"
echo "Start training of TigerSmall"
python3 finetune.py --dataset tiger --model_name  1811_OneOld_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant OneOld

echo "Start training of TübaSmall"
python3 finetune.py --dataset tüba --model_name  1811_OneOld_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant OneOld

echo "Start training of MergedSmall"
python3 finetune.py --dataset merged --model_name  1811_OneOld_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant OneOld

echo "Start training of TigerBase"
python3 finetune.py --dataset tiger --model_name  1811_OneOld_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant OneOld

echo "Start training of TübaBase"
python3 finetune.py --dataset tüba --model_name  1811_OneOld_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant OneOld

echo "Start training of MergedBase"
python3 finetune.py --dataset merged --model_name  1811_OneOld_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant OneOld

echo "============================================================="
echo "Start training of: One Old No No CCE"
echo "Start training of TigerSmall"
python3 finetune.py --dataset tiger --model_name  1811_NoNoCCE_OneOld_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1 --data_variant OneOld

echo "Start training of TübaSmall"
python3 finetune.py --dataset tüba --model_name  1811_NoNoCCE_OneOld_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1 --data_variant OneOld

echo "Start training of MergedSmall"
python3 finetune.py --dataset merged --model_name  1811_NoNoCCE_OneOld_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 1 --data_variant OneOld

echo "Start training of TigerBase"
python3 finetune.py --dataset tiger --model_name  1811_NoNoCCE_OneOld_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1 --data_variant OneOld

echo "Start training of TübaBase"
python3 finetune.py --dataset tüba --model_name  1811_NoNoCCE_OneOld_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1 --data_variant OneOld

echo "Start training of MergedBase"
python3 finetune.py --dataset merged --model_name  1811_NoNoCCE_OneOld_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 1 --data_variant OneOld

echo "============================================================="
echo "Start training of: One New"
echo "Start training of TigerSmall"
python3 finetune.py --dataset tiger --model_name  1811_OneNew_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant OneNew

echo "Start training of TübaSmall"
python3 finetune.py --dataset tüba --model_name  1811_OneNew_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant OneNew

echo "Start training of MergedSmall"
python3 finetune.py --dataset merged --model_name  1811_OneNew_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant OneNew

echo "Start training of TigerBase"
python3 finetune.py --dataset tiger --model_name  1811_OneNew_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant OneNew

echo "Start training of TübaBase"
python3 finetune.py --dataset tüba --model_name  1811_OneNew_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant OneNew

echo "Start training of MergedBase"
python3 finetune.py --dataset merged --model_name  1811_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant OneNew

echo "============================================================="
echo "Start training of: All Old"
echo "Start training of TigerSmall"
python3 finetune.py --dataset tiger --model_name  1811_AllOld_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllOld

echo "Start training of TübaSmall"
python3 finetune.py --dataset tüba --model_name  1811_AllOld_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllOld

echo "Start training of MergedSmall"
python3 finetune.py --dataset merged --model_name  1811_AllOld_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllOld

echo "Start training of TigerBase"
python3 finetune.py --dataset tiger --model_name  1811_AllOld_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllOld

echo "Start training of TübaBase"
python3 finetune.py --dataset tüba --model_name  1811_AllOld_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllOld

echo "Start training of MergedBase"
python3 finetune.py --dataset merged --model_name  1811_AllOld_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllOld

echo "============================================================="
echo "Start training of: All New"
echo "Start training of TigerSmall"
python3 finetune.py --dataset tiger --model_name  1811_AllNew_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "Start training of TübaSmall"
python3 finetune.py --dataset tüba --model_name  1811_AllNew_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "Start training of MergedSmall"
python3 finetune.py --dataset merged --model_name  1811_AllNew_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "Start training of TigerBase"
python3 finetune.py --dataset tiger --model_name  1811_AllNew_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "Start training of TübaBase"
python3 finetune.py --dataset tüba --model_name  1811_AllNew_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "Start training of MergedBase"
python3 finetune.py --dataset merged --model_name  1811_AllNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New Mixed"

echo "Start training of MergedSmall"
python3 finetune.py --dataset mergedMixed --model_name  1811_AllNew_MixedMergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "Start training of MergedBase"
python3 finetune.py --dataset mergedMixed --model_name  1811_AllNew_MixedMergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New 5050"
python3 finetune.py --dataset mergedFair --model_name  1811_AllNew_FairMergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0 --data_variant AllNew

echo "Start training of MergedBase"
python3 finetune.py --dataset mergedFair --model_name  1811_AllNew_FairMergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0 --data_variant AllNew

echo "====================== EVAL ================================="
echo "============================================================="
echo "Start evaluation of: Exact Match / One Old"
python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerSmall  --corpus eval --prefix 1811_OneOld_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerSmall  --corpus tuba --prefix 1811_OneOld_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerSmall  --corpus tiger --prefix 1811_OneOld_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerSmall  --corpus merged --prefix 1811_OneOld_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaSmall  --corpus tuba --prefix 1811_OneOld_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaSmall  --corpus tiger --prefix 1811_OneOld_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaSmall  --corpus eval --prefix 1811_OneOld_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaSmall  --corpus merged --prefix 1811_OneOld_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedSmall  --corpus eval --prefix 1811_OneOld_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedSmall  --corpus tuba --prefix 1811_OneOld_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedSmall  --corpus tiger --prefix 1811_OneOld_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedSmall  --corpus merged --prefix 1811_OneOld_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaBase  --corpus eval --prefix 1811_OneOld_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaBase  --corpus tuba --prefix 1811_OneOld_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaBase  --corpus tiger --prefix 1811_OneOld_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TübaBase  --corpus merged --prefix 1811_OneOld_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerBase  --corpus tuba --prefix 1811_OneOld_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerBase  --corpus tiger --prefix 1811_OneOld_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerBase  --corpus eval --prefix 1811_OneOld_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_TigerBase  --corpus merged --prefix 1811_OneOld_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedBase  --corpus eval --prefix 1811_OneOld_MeBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedBase  --corpus tuba --prefix 1811_OneOld_MeBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedBase  --corpus tiger --prefix 1811_OneOld_MeBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneOld_MergedBase  --corpus merged --prefix 1811_OneOld_MeBas

echo "============================================================="
echo "Start evaluation of: One Old No No CCE"
python3 test_model.py --checkpoint /home/marisa/models/1811_NoNoCCE_OneOld_TigerSmall  --corpus merged --prefix 1811_NoNoCCE_OneOld_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_NoNoCCE_OneOld_TübaSmall  --corpus merged --prefix 1811_NoNoCCE_OneOld_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_NoNoCCE_OneOld_MergedSmall  --corpus merged --prefix 1811_NoNoCCE_OneOld_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_NoNoCCE_OneOld_TigerBase  --corpus merged --prefix 1811_NoNoCCE_OneOld_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_NoNoCCE_OneOld_TübaBase  --corpus merged --prefix 1811_NoNoCCE_OneOld_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_NoNoCCE_OneOld_MergedBase  --corpus merged --prefix 1811_NoNoCCE_OneOld_MeBas

echo "============================================================="
echo "Start evaluation of: One New"
python3 test_model.py --checkpoint /home/marisa/models/1811_OneNew_TigerSmall  --corpus merged --prefix 1811_OneNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneNew_TübaSmall  --corpus merged --prefix 1811_OneNew_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneNew_MergedSmall  --corpus merged --prefix 1811_OneNew_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_OneNew_TigerBase  --corpus merged --prefix 1811_OneNew_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneNew_TübaBase  --corpus merged --prefix 1811_OneNew_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_OneNew_MergedBase  --corpus merged --prefix 1811_OneNew_MeBas

echo "============================================================="
echo "Start evaluation of: All Old"
python3 test_model.py --checkpoint /home/marisa/models/1811_AllOld_TigerSmall  --corpus merged --prefix 1811_AllOld_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllOld_TübaSmall  --corpus merged --prefix 1811_AllOld_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllOld_MergedSmall  --corpus merged --prefix 1811_AllOld_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllOld_TigerBase  --corpus merged --prefix 1811_AllOld_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllOld_TübaBase  --corpus merged --prefix 1811_AllOld_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllOld_MergedBase  --corpus merged --prefix 1811_AllOld_MeBas

echo "============================================================="
echo "Start evaluation of: All New"
python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TigerSmall  --corpus merged --prefix 1811_AllNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TübaSmall  --corpus merged --prefix 1811_AllNew_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_MergedSmall  --corpus merged --prefix 1811_AllNew_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TigerBase  --corpus merged --prefix 1811_AllNew_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TübaBase  --corpus merged --prefix 1811_AllNew_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_MergedBase  --corpus merged --prefix 1811_AllNew_MeBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TigerSmall  --corpus merged50 --prefix 1811_AllNew50_TiSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TübaSmall  --corpus merged50 --prefix 1811_AllNew50_TuSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_MergedSmall  --corpus merged50 --prefix 1811_AllNew50_MeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TigerBase  --corpus merged50 --prefix 1811_AllNew50_TiBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_TübaBase  --corpus merged50 --prefix 1811_AllNew50_TuBas

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_MergedBase  --corpus merged50 --prefix 1811_AllNew50_MeBas

echo "============================================================="
echo "Start evaluation of: All New Mixed"
python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_MixedMergedSmall  --corpus merged --prefix 1811_AllNew_MixMeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_MixedMergedBase  --corpus merged --prefix 1811_AllNew_MixMeBas

echo "============================================================="
echo "Start evaluation of: All New 5050"
python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_FairMergedSmall  --corpus merged --prefix 1811_AllNew_FaiMeSm

python3 test_model.py --checkpoint /home/marisa/models/1811_AllNew_FairMergedBase  --corpus merged --prefix 1811_AllNew_FaiMeBas

echo "DONE DONE DONE!"
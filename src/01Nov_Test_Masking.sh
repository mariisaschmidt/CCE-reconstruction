echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  01nov_TEST__OneOld_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0

echo "Start training of TübaSmall"

python3 finetune.py --dataset tüba --model_name  Test_OneOld_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0

echo "Start training of MergedSmall"

python3 finetune.py --dataset merged --model_name  Test_OneOld_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small --remove_no_cce 0

# echo "Start training of TigerBase"

# python3 finetune.py --dataset tiger --model_name  01nov_TEST__OneOld_TigerBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0

# echo "Start training of TübaBase"

# python3 finetune.py --dataset tüba --model_name  01nov_TEST__OneOld_TübaBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0

# echo "Start training of MergedBase"

# python3 finetune.py --dataset merged --model_name  01nov_TEST__OneOld_MergedBase  --pretrained_model /home/marisa/models/Aug25Base --remove_no_cce 0

echo "Done with training!"

# echo "Starting evaluation!"

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TigerSmall  --corpus eval --prefix 01nov_TEST__OneOld_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TigerSmall  --corpus tuba --prefix 01nov_TEST__OneOld_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TigerSmall  --corpus tiger --prefix 01nov_TEST__OneOld_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TübaSmall  --corpus tuba --prefix 01nov_TEST__OneOld_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TübaSmall  --corpus tiger --prefix 01nov_TEST__OneOld_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TübaSmall  --corpus eval --prefix 01nov_TEST__OneOld_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_MergedSmall  --corpus eval --prefix 01nov_TEST__OneOld_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_MergedSmall  --corpus tuba --prefix 01nov_TEST__OneOld_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_MergedSmall  --corpus tiger --prefix 01nov_TEST__OneOld_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TübaBase  --corpus eval --prefix 01nov_TEST__OneOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TübaBase  --corpus tuba --prefix 01nov_TEST__OneOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TübaBase  --corpus tiger --prefix 01nov_TEST__OneOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TigerBase  --corpus tuba --prefix 01nov_TEST__OneOld_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TigerBase  --corpus tiger --prefix 01nov_TEST__OneOld_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_TigerBase  --corpus eval --prefix 01nov_TEST__OneOld_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_MergedBase  --corpus eval --prefix 01nov_TEST__OneOld_MeBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_MergedBase  --corpus tuba --prefix 01nov_TEST__OneOld_MeBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST__OneOld_MergedBase  --corpus tiger --prefix 01nov_TEST__OneOld_MeBas

# echo "Done with evaluation!"
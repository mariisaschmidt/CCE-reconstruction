echo "Start training of TigerSmall"

python3 finetune.py --dataset tiger --model_name  01nov_TEST_OneNew_TigerSmall  --pretrained_model /home/marisa/models/Aug25Small  

# echo "Start training of TübaSmall"

# python3 finetune.py --dataset tüba --model_name  Test_OneNew_TübaSmall  --pretrained_model /home/marisa/models/Aug25Small  

# echo "Start training of MergedSmall"

# python3 finetune.py --dataset merged --model_name  Test_OneNew_MergedSmall  --pretrained_model /home/marisa/models/Aug25Small  

# echo "Start training of TigerBase"

# python3 finetune.py --dataset tiger --model_name  01nov_TEST_OneNew_TigerBase  --pretrained_model /home/marisa/models/Aug25Base  

# echo "Start training of TübaBase"

# python3 finetune.py --dataset tüba --model_name  01nov_TEST_OneNew_TübaBase  --pretrained_model /home/marisa/models/Aug25Base  

# echo "Start training of MergedBase"

# python3 finetune.py --dataset merged --model_name  01nov_TEST_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base  

echo "Done with training!"

echo "Starting evaluation!"

python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TigerSmall  --corpus eval --prefix 01nov_TEST_OneNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TigerSmall  --corpus tuba --prefix 01nov_TEST_OneNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TigerSmall  --corpus tiger --prefix 01nov_TEST_OneNew_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TübaSmall  --corpus tuba --prefix 01nov_TEST_OneNew_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TübaSmall  --corpus tiger --prefix 01nov_TEST_OneNew_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TübaSmall  --corpus eval --prefix 01nov_TEST_OneNew_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_MergedSmall  --corpus eval --prefix 01nov_TEST_OneNew_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_MergedSmall  --corpus tuba --prefix 01nov_TEST_OneNew_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_MergedSmall  --corpus tiger --prefix 01nov_TEST_OneNew_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TübaBase  --corpus eval --prefix 01nov_TEST_OneNew_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TübaBase  --corpus tuba --prefix 01nov_TEST_OneNew_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TübaBase  --corpus tiger --prefix 01nov_TEST_OneNew_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TigerBase  --corpus tuba --prefix 01nov_TEST_OneNew_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TigerBase  --corpus tiger --prefix 01nov_TEST_OneNew_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_TigerBase  --corpus eval --prefix 01nov_TEST_OneNew_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_MergedBase  --corpus eval --prefix 01nov_TEST_OneNew_MeBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_MergedBase  --corpus tuba --prefix 01nov_TEST_OneNew_MeBas

# python3 test_model.py --checkpoint /home/marisa/models/01nov_TEST_OneNew_MergedBase  --corpus tiger --prefix 01nov_TEST_OneNew_MeBas

# echo "Done with evaluation!"
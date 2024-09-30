echo "Start training of optimized model"

python3 train_optimized.py --dataset merged --model_name  30sep_BleuOptim_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Start evaluation of optimized model"

python3 test_model.py --checkpoint /home/marisa/models/30sep_BleuOptim_OneNew_MergedBase  --corpus eval --prefix 30sep_BleuOpti_OneNew_MeBas

python3 test_model.py --checkpoint /home/marisa/models/30sep_BleuOptim_OneNew_MergedBase  --corpus tiger --prefix 30sep_BleuOpti_OneNew_MeBas

python3 test_model.py --checkpoint /home/marisa/models/30sep_BleuOptim_OneNew_MergedBase  --corpus tuba --prefix 30sep_BleuOpti_OneNew_MeBas

echo "Done :)"
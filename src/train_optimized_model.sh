echo "Start training of optimized model"

python3 train_optimized.py --dataset merged --model_name  26sep_Optimized_OneNew_MergedBase  --pretrained_model /home/marisa/models/Aug25Base

echo "Start evaluation of optimized model"

python3 test_model.py --checkpoint /home/marisa/models/26sep_Optimized_OneNew_MergedBase  --corpus eval --prefix 22sep_Opti_OneNew_MeBas

echo "Done :)"
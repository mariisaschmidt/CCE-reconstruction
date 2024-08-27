echo "Starting evaluation!"

python3 test_model.py --checkpoint /home/marisa/models/27cleanTigerSmall  --corpus eval --prefix 27aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/27cleanTigerSmall  --corpus tuba --prefix 27aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/27cleanTigerSmall  --corpus tiger --prefix 27aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/27cleanTübaBase  --corpus eval --prefix 27aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/27cleanTübaBase  --corpus tuba --prefix 27aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/27cleanTübaBase  --corpus tiger --prefix 27aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/27cleanTigerBase  --corpus tuba --prefix 27aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/27cleanTigerBase  --corpus tiger --prefix 27aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/27cleanTigerBase  --corpus eval --prefix 27aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/27cleanTübaSmall  --corpus tuba --prefix 27aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/27cleanTübaSmall  --corpus tiger --prefix 27aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/27cleanTübaSmall  --corpus eval --prefix 27aug_TuSm

echo "Done with evaluation!"
echo "Starting evaluation!"

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall27  --corpus eval --prefix 27aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall27  --corpus tuba --prefix 27aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall27  --corpus tiger --prefix 27aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase27  --corpus eval --prefix 27aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase27  --corpus tuba --prefix 27aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase27  --corpus tiger --prefix 27aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase27  --corpus tuba --prefix 27aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase27  --corpus tiger --prefix 27aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase27  --corpus eval --prefix 27aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall27  --corpus tuba --prefix 27aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall27  --corpus tiger --prefix 27aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall27  --corpus eval --prefix 27aug_TuSm

echo "Done with evaluation!"
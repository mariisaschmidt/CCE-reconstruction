echo "Starting evaluation! ...again"

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall23  --corpus eval --prefix 24aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall23  --corpus tuba --prefix 24aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall23  --corpus tiger --prefix 24aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase23  --corpus eval --prefix 24aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase23  --corpus tuba --prefix 24aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase23  --corpus tiger --prefix 24aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase23  --corpus tuba --prefix 24aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase23  --corpus tiger --prefix 24aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase23  --corpus eval --prefix 24aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall23  --corpus tuba --prefix 24aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall23  --corpus tiger --prefix 24aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall23  --corpus eval --prefix 24aug_TuSm

echo "Done with evaluation!"
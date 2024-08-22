echo "Starting evaluation! ...again"

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall23 --corpus eval --prefix aug23_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall23 --corpus tuba --prefix aug23_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall23 --corpus tiger --prefix aug23_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase23 --corpus eval --prefix aug23_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase23 --corpus tuba --prefix aug23_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase23 --corpus tiger --prefix aug23_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase23 --corpus tuba --prefix aug23_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase23 --corpus tiger --prefix aug23_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase23 --corpus eval --prefix aug23_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall23 --corpus tuba --prefix aug23_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall23 --corpus tiger --prefix aug23_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall23 --corpus eval --prefix aug23_TuSm

echo "Done with evaluation!"
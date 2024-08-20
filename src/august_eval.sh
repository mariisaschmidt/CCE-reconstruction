echo "Starting evaluation!"

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall --corpus eval --prefix aug21_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall --corpus tuba --prefix aug21_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall --corpus tiger --prefix aug21_TiSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase --corpus eval --prefix aug21_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase --corpus tuba --prefix aug21_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase --corpus tiger --prefix aug21_TuBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase --corpus tuba --prefix aug21_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase --corpus tiger --prefix aug21_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase --corpus eval --prefix aug21_TiBas

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall --corpus tuba --prefix aug21_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall --corpus tiger --prefix aug21_TuSm

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall --corpus eval --prefix aug21_TuSm

echo "Done with evaluation!"
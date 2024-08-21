echo "Starting evaluation! ...again"

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall21 --corpus eval --prefix aug21_TiSmNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall21 --corpus tuba --prefix aug21_TiSmNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerSmall21 --corpus tiger --prefix aug21_TiSmNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase21 --corpus eval --prefix aug21_TuBasNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase21 --corpus tuba --prefix aug21_TuBasNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaBase21 --corpus tiger --prefix aug21_TuBasNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase21 --corpus tuba --prefix aug21_TiBasNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase21 --corpus tiger --prefix aug21_TiBasNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTigerBase21 --corpus eval --prefix aug21_TiBasNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall21 --corpus tuba --prefix aug21_TuSmNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall21 --corpus tiger --prefix aug21_TuSmNEW

python3 test_model.py --checkpoint /home/marisa/models/cleanTübaSmall21 --corpus eval --prefix aug21_TuSmNEW

echo "Done with evaluation!"
echo "Starting evaluation!"

python3 test_model.py --checkpoint /home/marisa/models/28cleanTigerSmall  --corpus eval --prefix 30aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/28cleanTigerSmall  --corpus tuba --prefix 30aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/28cleanTigerSmall  --corpus tiger --prefix 30aug_TiSm

python3 test_model.py --checkpoint /home/marisa/models/28cleanTübaBase  --corpus eval --prefix 30aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/28cleanTübaBase  --corpus tuba --prefix 30aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/28cleanTübaBase  --corpus tiger --prefix 30aug_TuBas

python3 test_model.py --checkpoint /home/marisa/models/28cleanTigerBase  --corpus tuba --prefix 30aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/28cleanTigerBase  --corpus tiger --prefix 30aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/28cleanTigerBase  --corpus eval --prefix 30aug_TiBas

python3 test_model.py --checkpoint /home/marisa/models/28cleanTübaSmall  --corpus tuba --prefix 30aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/28cleanTübaSmall  --corpus tiger --prefix 30aug_TuSm

python3 test_model.py --checkpoint /home/marisa/models/28cleanTübaSmall  --corpus eval --prefix 30aug_TuSm

echo "Done with evaluation!"
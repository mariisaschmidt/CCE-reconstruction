echo "Starting evaluation!"

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall  --corpus eval --prefix 12sep_AllNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall  --corpus tuba --prefix 12sep_AllNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall  --corpus tiger --prefix 12sep_AllNew_TiSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall  --corpus tuba --prefix 12sep_AllNew_TuSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall  --corpus tiger --prefix 12sep_AllNew_TuSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall  --corpus eval --prefix 12sep_AllNew_TuSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_MergedSmall  --corpus eval --prefix 12sep_AllNew_MeSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_MergedSmall  --corpus tuba --prefix 12sep_AllNew_MeSm

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_MergedSmall  --corpus tiger --prefix 12sep_AllNew_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TübaBase  --corpus eval --prefix 12sep_OneOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TübaBase  --corpus tuba --prefix 12sep_OneOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TübaBase  --corpus tiger --prefix 12sep_OneOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TigerBase  --corpus tuba --prefix 12sep_OneOld_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TigerBase  --corpus tiger --prefix 12sep_OneOld_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TigerBase  --corpus eval --prefix 12sep_OneOld_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_MergedBase  --corpus eval --prefix 12sep_OneOld_MeBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_MergedBase  --corpus tuba --prefix 12sep_OneOld_MeBas

# python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_MergedBase  --corpus tiger --prefix 12sep_OneOld_MeBas

echo "Done with evaluation!"
echo "Starting evaluation!"

# python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall  --corpus eval --prefix 12sep_AllNew_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall  --corpus tuba --prefix 12sep_AllNew_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall  --corpus tiger --prefix 12sep_AllNew_TiSm

# python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall  --corpus tuba --prefix 12sep_AllNew_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall  --corpus tiger --prefix 12sep_AllNew_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall  --corpus eval --prefix 12sep_AllNew_TuSm

# python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedSmall  --corpus eval --prefix 21sep_OneNew_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/21sep_AOneNew_MergedSmall  --corpus tuba --prefix 21sep_AOneNew_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedSmall  --corpus tiger --prefix 21sep_AllOld_MeSm

# python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_TübaBase  --corpus eval --prefix 21sep_AllOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_TübaBase  --corpus tuba --prefix 21sep_AllOld_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_TübaBase  --corpus tiger --prefix 22sep_OneNew_TuBas

# python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_TigerBase  --corpus tuba --prefix 22sep_OneNew_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_TigerBase  --corpus tiger --prefix 22sep_OneNew_TiBas

# python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_TigerBase  --corpus eval --prefix 22sep_OneNew_TiBas

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedBase  --corpus eval --prefix 21sep_AllOld_MeBas

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedBase  --corpus tuba --prefix 21sep_AllOld_MeBas

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedBase  --corpus tiger --prefix 21sep_AllOld_MeBas

echo "Done with evaluation!"
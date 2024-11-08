echo "Starting evaluation of all models with Merged Testdata!"

python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TübaSmall --corpus merged --prefix 0811_OneOld_TübaSmall 

python3 test_model.py --checkpoint /home/marisa/models/12sep_OneOld_TigerSmall --corpus merged --prefix 0811_OneOld_TigerSmall 

python3 test_model.py --checkpoint /home/marisa/models/21sep_OneOld_MergedSmall --corpus merged --prefix 0811_OneOld_MergedSmall 

python3 test_model.py --checkpoint /home/marisa/models/21sep_OneOld_TigerBase  --corpus merged --prefix 0811_OneOld_TigerBase 

python3 test_model.py --checkpoint /home/marisa/models/21sep_OneOld_TübaBase --corpus merged --prefix 0811_OneOld_TübaBase 

python3 test_model.py --checkpoint /home/marisa/models/21sep_OneOld_MergedBase --corpus merged --prefix 0811_OneOld_MergedBase

echo "Done with One Old"
              
python3 test_model.py --checkpoint /home/marisa/models/01oct_NoNoCCE_OneOld_MergedBase --corpus merged --prefix 0811_NoNoCCE_OneOld_MergedBase 

python3 test_model.py --checkpoint /home/marisa/models/01oct_NoNoCCE_OneOld_MergedSmall --corpus merged --prefix 0811_NoNoCCE_OneOld_MergedSmall

python3 test_model.py --checkpoint /home/marisa/models/01oct_NoNoCCE_OneOld_TigerBase --corpus merged --prefix 0811_NoNoCCE_OneOld_TigerBase

python3 test_model.py --checkpoint /home/marisa/models/01oct_NoNoCCE_OneOld_TigerSmall --corpus merged --prefix 0811_NoNoCCE_OneOld_TigerSmall

python3 test_model.py --checkpoint /home/marisa/models/01oct_NoNoCCE_OneOld_TübaBase --corpus merged --prefix 0811_NoNoCCE_OneOld_TübaBase 

python3 test_model.py --checkpoint /home/marisa/models/01oct_NoNoCCE_OneOld_TübaSmall --corpus merged --prefix 0811_NoNoCCE_OneOld_TübaSmall 

echo "Done with One Old No No CCE"  

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedBase --corpus merged --prefix 0811_AllOld_MergedBase 

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_MergedSmall --corpus merged --prefix 0811_AllOld_MergedSmall  

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_TigerBase --corpus merged --prefix 0811_AllOld_TigerBase 

python3 test_model.py --checkpoint /home/marisa/models/21sep_AllOld_TübaBase  --corpus merged --prefix 0811_AllOld_TübaBase 

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllOld_TigerSmall  --corpus merged --prefix 0811_AllOld_TigerSmall

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllOld_TübaSmall --corpus merged --prefix 0811_AllOld_TübaSmall

echo "Done with All Old"         

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TigerSmall --corpus merged --prefix 0811_AllNew_TigerSmall

python3 test_model.py --checkpoint /home/marisa/models/22sep_AllNew_MergedSmall --corpus merged --prefix 0811_AllNew_MergedSmall  

python3 test_model.py --checkpoint /home/marisa/models/22sep_AllNew_TigerBase --corpus merged --prefix 0811_AllNew_TigerBase 

python3 test_model.py --checkpoint /home/marisa/models/22sep_AllNew_TübaBase  --corpus merged --prefix 0811_AllNew_TübaBase 

python3 test_model.py --checkpoint /home/marisa/models/22sep_AllNew_MergedBase --corpus merged --prefix 0811_AllNew_MergedBase

python3 test_model.py --checkpoint /home/marisa/models/12sep_AllNew_TübaSmall --corpus merged --prefix 0811_AllNew_TübaSmall

echo "Done with All New"             
                  
python3 test_model.py --checkpoint /home/marisa/models/12sep_OneNew_TigerSmall --corpus merged --prefix 0811_OneNew_TigerSmall

python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_MergedSmall --corpus merged --prefix 0811_OneNew_MergedSmall  

python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_TigerBase --corpus merged --prefix 0811_OneNew_TigerBase 

python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_TübaBase --corpus merged --prefix 0811_OneNew_TübaBase 

python3 test_model.py --checkpoint /home/marisa/models/22sep_OneNew_MergedBase --corpus merged --prefix 0811_OneNew_MergedBase

python3 test_model.py --checkpoint /home/marisa/models/12sep_OneNew_TübaSmall --corpus merged --prefix 0811_OneNew_TübaSmall

echo "Done with One New"                   
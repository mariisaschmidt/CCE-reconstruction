echo "This script will get information about the average sentence length in each dataset."

cd ../src

echo "TSD models"
echo "TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant TSD

echo "TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant TSD

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant TSD

echo "============================================================="

echo "Exact Match / One Old"
echo "TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant OneOld

echo "TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant OneOld

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant OneOld

echo "============================================================="
echo "One Old No No CCE"
echo "TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 1 --data_variant OneOld

echo "TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 1 --data_variant OneOld

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 1 --data_variant OneOld

echo "============================================================="
echo "One New"
echo "TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant OneNew

echo "TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant OneNew

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant OneNew

echo "============================================================="
echo "All Old"
echo "TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant AllOld

echo "TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant AllOld

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant AllOld

echo "============================================================="
echo "All New"
echo "TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant AllNew

echo "TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant AllNew

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "All New Mixed"

echo "MergedSmall"
python3 get_avg_sentence_length.py --dataset mergedMixed --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "All New 5050"
python3 get_avg_sentence_length.py --dataset mergedFair --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "All New 5050 Large"
python3 get_avg_sentence_length.py --dataset mergedFairLarge --remove_no_cce 0 --data_variant AllNew


echo "xxxxxxxxxxxxxxxxxxxxxxxx"
echo "Info about Eval data"
python3 analyze_testdata.py 

cd ../
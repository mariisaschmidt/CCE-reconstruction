cd ../src

echo "Start TSD"
echo "Start TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant TSD

echo "Start TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant TSD

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant TSD

echo "============================================================="

echo "Start training of: Exact Match / One Old"
echo "Start TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant OneOld

echo "Start TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant OneOld

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant OneOld

echo "============================================================="
echo "Start training of: One Old No No CCE"
echo "Start TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 1 --data_variant OneOld

echo "Start TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 1 --data_variant OneOld

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 1 --data_variant OneOld

echo "============================================================="
echo "Start training of: One New"
echo "Start TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant OneNew

echo "Start TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant OneNew

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant OneNew

echo "============================================================="
echo "Start training of: All Old"
echo "Start TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant AllOld

echo "Start TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant AllOld

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant AllOld

echo "============================================================="
echo "Start training of: All New"
echo "Start TigerSmall"
python3 get_avg_sentence_length.py --dataset tiger --remove_no_cce 0 --data_variant AllNew

echo "Start TübaSmall"
python3 get_avg_sentence_length.py --dataset tüba --remove_no_cce 0 --data_variant AllNew

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset merged --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New Mixed"

echo "Start MergedSmall"
python3 get_avg_sentence_length.py --dataset mergedMixed --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New 5050"
python3 get_avg_sentence_length.py --dataset mergedFair --remove_no_cce 0 --data_variant AllNew

echo "============================================================="
echo "Start training of: All New 5050 Large"
python3 get_avg_sentence_length.py --dataset mergedFairLarge --remove_no_cce 0 --data_variant AllNew


echo "xxxxxxxxxxxxxxxxxxxxxxxx"
echo "Info about Eval data"
python3 analyze_testdata.py 

cd ../
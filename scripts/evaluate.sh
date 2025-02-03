read -p "Which model do you want to evaluate? (Path): " modelname
cd ../src
python3 test_model.py --checkpoint modelname
cd ..
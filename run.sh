print_help(){
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "prepare_g4 - prepares the G4 corpus"
    echo "prepare_tuba - prepares the TüBa corpus"
    echo "prepare_tiger - prepares the TIGER corpus"
    echo "train - asks for some inputs and trains a model based on them"
    echo "evaluate - evaluate a trained model"
}

prepare_g4(){
    cd scripts
    chmod +x convert_file.sh
    python3 download_files.py
    mkdir base_files
    mv *.jsonl base_files/
    mkdir correct_files
    python3 write_correct_files.py
    rm -r base_files
    python3 get_translation_pairs.py
    rm -r correct_files
    mv "de_de_pairs.jsonl" "../data/"
    cd ..
}

prepare_tuba(){
    cd src
    mkdir tmp_json
    python3 get_ellipsis_data_tüba.py
    rm -r tmp_json
    mv "tüba_train.jsonl" "../data/"
    mv "tüba_test.jsonl" "../data/"
    cd ..
}

prepare_tiger(){
    cd src
    mkdir tmp_json2
    python3 get_ellipsis_data_tiger.py
    rm -r tmp_json2
    mv "tiger_train.jsonl" "../data/"
    mv "tiger_test.jsonl" "../data/"
    cd ..
}

train(){
    cd scripts
    chmod +x train.sh
    ./train.sh 
    cd ..
}

evaluate(){
    cd scripts
    chmod +x evaluate.sh
    ./evaluate.sh
    cd .. 
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_help
    return 0
fi

case "$1" in
    "prepare_g4")
        prepare_g4
        ;;
    "train")
        train
        ;;
    "evaluate")
        evaluate
        ;;
    "prepare_tuba")
        prepare_tuba
        ;;
    "prepare_tiger")
        prepare_tiger
        ;;
    *)
        echo "Unknown option: $1"
        print_help
        exit 1
        ;;
esac
print_help(){
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "create_venv - creates virtual environment called cce"
    echo "activate_venv - activates the created virtual environment"
    echo "prepare_g4 - prepares the G4 corpus"
    echo "prepare_tuba - prepares the TüBa corpus"
    echo "prepare_tiger - prepares the TIGER corpus"
    echo "train - asks for some inputs and trains a model based on them"
    echo "evaluate - evaluate a trained model"
}

check_virtualenv() {
    if ! command -v virtualenv &> /dev/null; then
        echo "virtualenv is not installed. Installing..."
        python3 -m pip install --user virtualenv
        echo "virtualenv installation complete."
    fi
}

create_venv() {
    # Check if virtualenv is installed, if not, install it
    check_virtualenv
    
    local env_name="cce"

    if [ -d "$env_name" ]; then
        echo "Virtual environment '$env_name' already exists. Aborting."
        return 1
    fi

    python3 -m venv "$env_name"
    source "./$env_name/bin/activate"
    pip install -U pip

    echo "Installing some requirements..."
    pip install transformers
    pip install datasets
    pip install tokenizers
    pip install pytorch
    pip install matplotlib
    pip install numpy
    pip install tqdm
    pip install argparse
    pip install nltk

    echo "Please run: nltk.download("punkt") once!"
    echo "Done!"
}

activate_venv() {
    local env_name="cce"

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create' to create it."
        return 1
    fi

    source "./$env_name/bin/activate"
}

prepare_g4(){
    activate_venv
    chmod +x convert_file.sh
    python3 download_files.py
    mkdir base_files
    mv *.jsonl base_files/
    mkdir correct_files
    python3 write_correct_files.py
    rm -r base_files
    python3 get_translation_pairs.py
    rm -r correct_files
}

prepare_tuba(){
    mkdir tmp_json
    python3 get_ellipsis_data_tüba.py
    rm -r tmp_json
}

prepare_tiger(){
    mkdir tmp_json2
    python3 get_ellipsis_data_tiger.py
    rm -r tmp_json2
}

train(){
    cd src
    chmod +x train.sh
    ./train.sh 
}

evaluate(){
    cd src
    chmod +x evaluate.sh
    ./evaluate.sh
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_help
    return 0
fi

case "$1" in
    "create_venv")
        create_venv
        ;;
    "activate_venv")
        activate_venv
        ;;
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
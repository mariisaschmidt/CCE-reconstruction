from datasets import load_dataset, Dataset, concatenate_datasets
import nltk
import numpy as np
import argparse
import os 
import random
from collections import defaultdict

sentence_counts = defaultdict(int)

# This function add one of three prefixes to a sentence 
def add_prefix_to_duplicates(example):
    global sentence_counts

    sentence = example["Original sentence"]
    gold_standard = example["Canonical form"]

    prefixe = ["Er sagt:", "Sie erzählt:", "Folgendes wird berichtet:"]
    sentence_counts[sentence] += 1
    prefix = random.choice(prefixe)

    modified_sentence = prefix + " " + sentence
    modified_gold_standard = prefix + " " + gold_standard

    modified_example = {key: value for key, value in example.items()}
    modified_example["Original sentence"] = modified_sentence
    modified_example["Canonical form"] = modified_gold_standard
    
    return modified_example

# This function removes sentences that don't contain cce 
def filter_no_cce(example):
    return all((example[feature] == 0) or example[feature] == "0" for feature in feature_columns)

# This function is used to create a combined dataset that contains 50% data from dataset_small and 50% data from dataset_large
def balance_datasets(dataset_small, dataset_large, feature_columns):
    balanced_data = []
    
    for feature in feature_columns:
        # Anzahl der Sätze mit Feature = 1 im kleineren Datensatz
        small_subset = dataset_small.filter(lambda x: (x[feature] == 1 ) or (x[feature] == "1"))
        count = len(small_subset)
        #print(feature, count)
        
        # Zufällige Auswahl der gleichen Anzahl aus dem größeren Datensatz
        large_subset = dataset_large.filter(lambda x: (x[feature] == 1 ) or (x[feature] == "1"))
        sampled_large_subset = large_subset.shuffle(seed=42).select(range(min(count, len(large_subset))))
        
        # Kombinieren der Subsets
        balanced_data.append(small_subset)
        balanced_data.append(sampled_large_subset)
    
    # Anzahl der Sätze ohne CCE im kleineren Datensatz
    small_subset = dataset_small.filter(filter_no_cce)
    count = len(small_subset)
    print("NO CCE", count)
    
    # Zufällige Auswahl der gleichen Anzahl aus dem größeren Datensatz
    large_subset = dataset_large.filter(lambda x: (x[feature] == 0 ) or (x[feature] == "0"))
    sampled_large_subset = large_subset.shuffle(seed=42).select(range(min(count, len(large_subset))))
    
    # Kombinieren der Subsets
    balanced_data.append(small_subset)
    balanced_data.append(sampled_large_subset)
    
    # Alle balancierten Subsets zusammenfügen
    combined_dict = {
        key: sum((subset[key] for subset in balanced_data), []) for key in dataset_small.features
    }
    return Dataset.from_dict(combined_dict)

# Funktion zur Berechnung der Satzlängen
def get_sentence_lengths(text):
    sentences = nltk.sent_tokenize(text)  # Sätze aufteilen
    word_counts = [len(sentence.split()) for sentence in sentences]  # Wörter pro Satz zählen
    return word_counts

def get_avg_length(dataset, text_column):
    all_lengths = [length for example in dataset[text_column] for length in get_sentence_lengths(example)]
    average_length = np.mean(all_lengths)
    print(f"Durchschnittliche Anzahl Wörter pro Satz: {average_length:.4f} Wörter. \n")

if __name__ == '__main__':
    # since the vm has no UI, this code relies on command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--remove_no_cce", type=int) # should sentences without cce be removed? 1 = yes
    parser.add_argument("--data_variant", type=str) # which variant of the dataset should be used? AllOld, AllNew, OneOld, OneNew, TSD 
    args = parser.parse_args()

    if args.dataset:
        dataset_name = args.dataset
    else:
        print("You need to specify a dataset! Options are: tüba, tiger, g4")
        dataset_name = ""
    if args.remove_no_cce:
        removeNoCce = args.remove_no_cce
    else: 
        removeNoCce = 0
    if args.data_variant:
        data_variant = args.data_variant
    else: 
        data_variant = " "

    if dataset_name != "":
        if dataset_name == "tüba":
            if data_variant == "OneOld":
                train_data = os.path.expanduser("../data/CLEANED_OLD_tüba_train.jsonl")
            elif data_variant == "AllOld":
                train_data = os.path.expanduser("../data/CLEANED_ALL_OLD_tüba_train.jsonl")
            elif data_variant == "OneNew":
                train_data = os.path.expanduser("../data/CLEANED_ONE_NEW_tüba_train.jsonl")
            elif data_variant == "AllNew":
                train_data = os.path.expanduser("../data/CLEANED_tüba_train.jsonl")
            elif data_variant == "TSD":
                train_data = os.path.expanduser("../data/tüba_train.jsonl")
            train_dataset = load_dataset("json", data_files=train_data, split='train')
            if removeNoCce == 1:
                cols_to_check = ['BCR', 'FCR', 'Gapping', 'SGF']
                print(train_dataset.num_rows)
                train_dataset = train_dataset.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset.num_rows)
            print("Got train data")
            t = "Treebank-Sentence"
            g = "Reconstructed-Sentence"
        if dataset_name == "tiger":
            if data_variant == "OneOld":
                train_data = os.path.expanduser("../data/CLEANED_OLD_tiger_train.jsonl")
            elif data_variant == "AllOld":
                train_data = os.path.expanduser("../data/CLEANED_ALL_OLD_tiger_train.jsonl")
            elif data_variant == "OneNew":
                train_data = os.path.expanduser("../data/CLEANED_ONE_NEW_tiger_train.jsonl")
            elif data_variant == "AllNew":
                train_data = os.path.expanduser("../data/CLEANED_tiger_train.jsonl")
            elif data_variant == "TSD":
                train_data = os.path.expanduser("../data/tiger_train.jsonl")
            train_dataset = load_dataset("json", data_files=train_data, split='train')
            if removeNoCce == 1:
                cols_to_check = ['BCR', 'FCR', 'Gapping', 'SGF']
                print(train_dataset.num_rows)
                train_dataset = train_dataset.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset.num_rows)
            print("Got train data")
            t = "Original sentence"
            g = "Canonical form"

        if dataset_name == "erwTiger":
            train_data = os.path.expanduser("../data/CLEANED_erweitert_tiger_train.jsonl")
            train_dataset = load_dataset("json", data_files=train_data, split='train')
            print("Got train data")
            t = "Original sentence"
            g = "Canonical form"

        if dataset_name == "merged":
            if data_variant == "OneOld":
                train_data1 = os.path.expanduser("../data/CLEANED_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_OLD_tüba_train.jsonl")
            elif data_variant == "AllOld":
                train_data1 = os.path.expanduser("../data/CLEANED_ALL_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ALL_OLD_tüba_train.jsonl")
            elif data_variant == "OneNew":
                train_data1 = os.path.expanduser("../data/CLEANED_ONE_NEW_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ONE_NEW_tüba_train.jsonl")
            elif data_variant == "AllNew":
                train_data1 = os.path.expanduser("../data/CLEANED_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_tüba_train.jsonl")
            elif data_variant == "TSD":
                train_data1 = os.path.expanduser("../data/tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/tüba_train.jsonl")

            train_dataset1 = load_dataset("json", data_files=train_data1, split='train')
            train_dataset2 = load_dataset("json", data_files=train_data2, split='train')
            train_dataset2 = train_dataset2.rename_column("Treebank-Sentence", "Original sentence")
            train_dataset2 = train_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
            if removeNoCce == 1:
                cols_to_check = ['BCR', 'FCR', 'Gapping', 'SGF']
                print(train_dataset1.num_rows)
                train_dataset1 = train_dataset1.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset1.num_rows)
                print(train_dataset2.num_rows)
                train_dataset2 = train_dataset2.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset2.num_rows)
            cols_to_remove1 = train_dataset1.column_names
            cols_to_remove2 = train_dataset2.column_names
            cols_to_remove2.remove("Original sentence")
            cols_to_remove2.remove("Canonical form")
            train_dataset2 = train_dataset2.remove_columns(cols_to_remove2)
            cols_to_remove1.remove("Original sentence")
            cols_to_remove1.remove("Canonical form")
            train_dataset1 = train_dataset1.remove_columns(cols_to_remove1)
            train_dataset = concatenate_datasets([train_dataset1, train_dataset2])
            print("Got train data")

            t = "Original sentence"
            g = "Canonical form"

        if dataset_name == "mergedMixed":
            if data_variant == "OneOld":
                train_data1 = os.path.expanduser("../data/CLEANED_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_OLD_tüba_train.jsonl")
            elif data_variant == "AllOld":
                train_data1 = os.path.expanduser("../data/CLEANED_ALL_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ALL_OLD_tüba_train.jsonl")
            elif data_variant == "OneNew":
                train_data1 = os.path.expanduser("../data/CLEANED_ONE_NEW_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ONE_NEW_tüba_train.jsonl")
            elif data_variant == "AllNew":
                train_data1 = os.path.expanduser("../data/CLEANED_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_tüba_train.jsonl")

            train_dataset1 = load_dataset("json", data_files=train_data1, split='train')
            train_dataset2 = load_dataset("json", data_files=train_data2, split='train')
            train_dataset2 = train_dataset2.rename_column("Treebank-Sentence", "Original sentence")
            train_dataset2 = train_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
            if removeNoCce == 1:
                cols_to_check = ['BCR', 'FCR', 'Gapping', 'SGF']
                print(train_dataset1.num_rows)
                train_dataset1 = train_dataset1.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset1.num_rows)
                print(train_dataset2.num_rows)
                train_dataset2 = train_dataset2.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset2.num_rows)
            cols_to_remove1 = train_dataset1.column_names
            cols_to_remove2 = train_dataset2.column_names
            cols_to_remove2.remove("Original sentence")
            cols_to_remove2.remove("Canonical form")
            train_dataset2 = train_dataset2.remove_columns(cols_to_remove2)
            cols_to_remove1.remove("Original sentence")
            cols_to_remove1.remove("Canonical form")
            train_dataset1 = train_dataset1.remove_columns(cols_to_remove1)
            train_dataset = concatenate_datasets([train_dataset1, train_dataset2])
            print("Got train data")

            train_dataset = train_dataset.shuffle(seed=3)
            #dataset.save_to_disk("MixedMergedTrainDataset")
            
            t = "Original sentence"
            g = "Canonical form"

        if dataset_name == "mergedFair":
            if data_variant == "OneOld":
                train_data1 = os.path.expanduser("../data/CLEANED_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_OLD_tüba_train.jsonl")
            elif data_variant == "AllOld":
                train_data1 = os.path.expanduser("../data/CLEANED_ALL_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ALL_OLD_tüba_train.jsonl")
            elif data_variant == "OneNew":
                train_data1 = os.path.expanduser("../data/CLEANED_ONE_NEW_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ONE_NEW_tüba_train.jsonl")
            elif data_variant == "AllNew":
                train_data1 = os.path.expanduser("../data/CLEANED_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_tüba_train.jsonl")

            train_dataset1 = load_dataset("json", data_files=train_data1, split='train')
            train_dataset2 = load_dataset("json", data_files=train_data2, split='train')
            train_dataset2 = train_dataset2.rename_column("Treebank-Sentence", "Original sentence")
            train_dataset2 = train_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
            if removeNoCce == 1:
                cols_to_check = ['BCR', 'FCR', 'Gapping', 'SGF']
                print(train_dataset1.num_rows)
                train_dataset1 = train_dataset1.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset1.num_rows)
                print(train_dataset2.num_rows)
                train_dataset2 = train_dataset2.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset2.num_rows)
            
            # Balancierter Datensatz
            feature_columns = ['BCR', 'FCR', 'Gapping', 'SGF']
            train_dataset = balance_datasets(train_dataset2, train_dataset1, feature_columns)
            print(train_dataset)
            print("Got train data")

            train_dataset = train_dataset.shuffle(seed=3)
            #dataset.save_to_disk("BalancedMergedTrainDataset")
            
            t = "Original sentence"
            g = "Canonical form"

        if dataset_name == "mergedFairLarge":
            if data_variant == "OneOld":
                train_data1 = os.path.expanduser("../data/CLEANED_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_OLD_tüba_train.jsonl")
            elif data_variant == "AllOld":
                train_data1 = os.path.expanduser("../data/CLEANED_ALL_OLD_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ALL_OLD_tüba_train.jsonl")
            elif data_variant == "OneNew":
                train_data1 = os.path.expanduser("../data/CLEANED_ONE_NEW_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_ONE_NEW_tüba_train.jsonl")
            elif data_variant == "AllNew":
                train_data1 = os.path.expanduser("../data/CLEANED_tiger_train.jsonl")
                train_data2 = os.path.expanduser("../data/CLEANED_tüba_train.jsonl")

            train_dataset1 = load_dataset("json", data_files=train_data1, split='train')
            train_dataset2 = load_dataset("json", data_files=train_data2, split='train')
            train_dataset2 = train_dataset2.rename_column("Treebank-Sentence", "Original sentence")
            train_dataset2 = train_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
            if removeNoCce == 1:
                cols_to_check = ['BCR', 'FCR', 'Gapping', 'SGF']
                print(train_dataset1.num_rows)
                train_dataset1 = train_dataset1.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset1.num_rows)
                print(train_dataset2.num_rows)
                train_dataset2 = train_dataset2.filter(lambda row: not all(row[col] == "0" for col in cols_to_check))
                print(train_dataset2.num_rows)
            
            # Balancierter Datensatz
            feature_columns = ['BCR', 'FCR', 'Gapping', 'SGF']
            train_dataset = balance_datasets(train_dataset1, train_dataset2, feature_columns) # groß und klein tauschen für erweiterung
            train_dataset.map(add_prefix_to_duplicates)
            print(train_dataset)
            print("Got train data")

            train_dataset = train_dataset.shuffle(seed=3)

            t = "Original sentence"
            g = "Canonical form"
    
    dataset = train_dataset 
    # don't use sentences with length < 20
    dataset = dataset.filter(lambda example: len(example[t]) >= 20) 
    dataset = dataset.filter(lambda example: len(example[g]) >= 20)

    get_avg_length(dataset, g)

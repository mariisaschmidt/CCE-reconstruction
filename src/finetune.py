# the all-in-one script to finetune a huggingface ellipsis reconstruction model 

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np 
import os 
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from datasets import Dataset
from collections import defaultdict
import random
import sys

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

# Simple preprocessing function that adds a prefix to the input sentence and apploes tokenization to get the model inputs
def preprocess_function(examples):
    inputs = prefix + examples[t]
    targets = examples[g]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True, padding='longest', return_tensors='pt')
    return model_inputs

# Somehow there is a [] too much, so we remove that here
def correct_inputs_masks_labels(examples):
    examples['input_ids'] = examples['input_ids'][0] 
    examples['attention_mask'] = examples['attention_mask'][0] 
    examples['labels'] = examples['labels'][0] 
    return examples

# Simple postprocessing function
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# Compute the evaluation metrics based on the predictions
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]} # compute bleu score in this case

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens) # compute mean of length of the predicted sentences
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == '__main__':
    # since the vm has no UI, this code relies on command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--remove_no_cce", type=int) # should sentences without cce be removed? 1 = yes
    parser.add_argument("--data_variant", type=str) # which variant of the dataset should be used? AllOld, AllNew, OneOld, OneNew, TSD 
    args = parser.parse_args()

    if args.dataset:
        dataset_name = args.dataset
    else:
        print("You need to specify a dataset! Options are: tüba, tiger, g4")
        dataset_name = ""
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "de_de_llm"
    if args.remove_no_cce:
        removeNoCce = args.remove_no_cce
    else: 
        removeNoCce = 0
    if args.data_variant:
        data_variant = args.data_variant
    else: 
        data_variant = " "
    if args.pretrained_model:
        checkpoint = args.pretrained_model
    else:
        if dataset_name == "g4":
            checkpoint = "t5-base"
        else: 
            print("You need to provide a german llm for finetuning with ellipsis data!")
    
    # The following block of code just determines 
    #   - which dataset to load
    #   - if it has to be further processed
    #   - how the relevant columns are called
    #   - values for batchsize and number of epochs
    #   - task prefix 
    # based on the given command line arguments (end of this block is line 404)
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
            prefix = "reconstruct the ellipsis in this sentence: "
            batchsize = 4
            epochs = 10 #5
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
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10 #5
        if dataset_name == "erwTiger":
            train_data = os.path.expanduser("../data/CLEANED_erweitert_tiger_train.jsonl")
            train_dataset = load_dataset("json", data_files=train_data, split='train')
            print("Got train data")
            t = "Original sentence"
            g = "Canonical form"
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            if removeNoCce == 1:
                epochs = 10
                print("10 EPOCHS")
            else:
                epochs = 5
                print("5 EPOCHS")
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
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10 #5
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
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10 #5

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
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10 #5
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
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10
        if dataset_name == "ECBAE":
            # TIGER ERWEITERT + TüBa 
            # Ein Gold, Aktualisiert
            # Gemischt
            train_data1 = os.path.expanduser("../data/CLEANED_erweitert_tiger_train.jsonl") # Erweiterter TIGER
            train_data2 = os.path.expanduser("../data/CLEANED_tüba_train.jsonl") # TüBa

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
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10 #5

        if dataset_name == "g4":
            data = os.path.expanduser("../data/CLEANED_de_de_pairs.jsonl")
            train_dataset = load_dataset("json", data_files=data, split='train')
            prefix = "translate German to German: "
            t = "text"
            g = "gold_sentence"
            batchsize = 16
            epochs = 2
        
    print("Loaded Dataset!")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint) # load tokenizer for the model

    source_lang = "de"
    target_lang = "de"

    dataset = train_dataset 
    # don't use sentences with length < 20
    dataset = dataset.filter(lambda example: len(example[t]) >= 20) 
    dataset = dataset.filter(lambda example: len(example[g]) >= 20)

    print("Preprocess Data: ")
    tokenized_dataset = dataset.map(preprocess_function, batched=False) # preprocess

    print("Correct the outputs of preprocess: ")
    tokenized_dataset = tokenized_dataset.map(correct_inputs_masks_labels, batched=False)

    print("Create Train-Test-Split: ")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=3)

# UNCOMMENT ONLY IF YOU WANT TO SAVE THE DATASET AS CSV
    print(tokenized_dataset)
    ds_name = "csv_" + dataset_name + "_" + data_variant + "_" + str(removeNoCce)
    print("Dataset Name: ", ds_name)
    tokenized_dataset["train"].to_csv(ds_name + "_train.csv")
    tokenized_dataset["test"].to_csv(ds_name + "_test.csv")
    #sys.exit()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint) # load data collator

    metric = evaluate.load("bleu") 

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # set logging directory
    log_dir = os.path.expanduser("../models/" + model_name + "/logs")

    training_args = Seq2SeqTrainingArguments(
    output_dir=model_name,
    evaluation_strategy="epoch",
    logging_dir=log_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=batchsize,
    per_device_eval_batch_size=batchsize,
    weight_decay=0.01,
    save_strategy="no",
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=False #True, # set true when cuda available
    push_to_hub=False,
    generation_max_length=256,
    log_level="debug", # enable more logs
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Train Model: ")
    trainer.train()
    print("Saving Model ..")
    trainer.save_model(output_dir=os.path.expanduser("../models/" + model_name))
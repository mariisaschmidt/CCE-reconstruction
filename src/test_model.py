import re
import os 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import evaluate
from datasets import load_dataset, concatenate_datasets

def get_predictions(ds, sc):
    inputs = ds[sc]
    # inputs = [clean_sentence(p) for p in inputs]
    predictions = []
    for input in inputs:
        evaluation_input = (tokenizer.encode(input, return_tensors="pt"))
        evaluation_output = model.generate(evaluation_input, max_new_tokens=200)
        decoded = tokenizer.decode(evaluation_output[0])
        decoded = decoded[6:len(decoded)-4]
        decoded = decoded.replace("<unk>", "Oe")
        predictions.append(decoded)
    return predictions

def remove_brackets_and_suffix(sentence):
    # converts [word cce-type] to word
    cleaned_sentence = re.sub(r'\[([^\s\]]+)\s[^\]]+\]', r'\1', sentence)
    return cleaned_sentence

def add_one_space(sentence):
    return sentence + " "

def evaluate_model(file, bleu, exmatch, dataset, name, add_space):
    predictions = get_predictions(dataset, sent_col)
    golds = dataset[gold_col]
    if add_space:
        for i in range(0, len(predictions)):
            if golds[i].endswith(" "):
                predictions[i] = add_one_space(predictions[i])

    file.write("======================" + name + "============================== \n")
    for j in range(0,2): # define multiple evaluation loops
        for i in range(0, len(predictions)):
            if removeCCEType:
                predictions[i] = remove_brackets_and_suffix(predictions[i])
                golds[i] = remove_brackets_and_suffix(golds[i])
            if j == 0:
                file.write("====================== PRED VS GOLD ============================== \n")
                file.write("pred: " + predictions[i] + "\n")
                file.write("gold: " + golds[i] + "\n")
                ems = exmatch.compute(references=[golds[i]], predictions=[predictions[i]], ignore_case=True, ignore_punctuation=True)
                file.write("EM Score: " + str(ems["exact_match"]) + " Length: " + str(len(predictions[i])) + " vs " + str(len(golds[i])) + "\n")
        if j == 1:
            file.write("====================== EXACT MATCH ============================== \n")
            if(len(predictions) != 0):
                score = bleu.compute(predictions=predictions, references=golds)
                file.write("Bleu Score: " + str(score) + "\n")
                exact_matches = exmatch.compute(references=golds, predictions=predictions, ignore_case=True, ignore_punctuation=True)
                file.write("Exact Matches: " + str(exact_matches["exact_match"]) + "\n")
                file.write("\n \n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()

    removeCCEType = False

    if args.checkpoint:
        checkpoint = args.checkpoint
        if "Masked" in checkpoint:
            removeCCEType = True 
            print("Removing brackets and suffix from sentences!")
    else:
        print("You need to specify the path to the model checkpoint you want to load!")
        checkpoint = " "
    
    if args.corpus == "tuba":
        corpus = os.path.expanduser("../data/CLEANED_OLD_tüba_test.jsonl")
        #corpus = os.path.expanduser("../data/tüba_test.jsonl")
        sent_col = "Treebank-Sentence"
        gold_col = "Reconstructed-Sentence"
        add_space = False
    elif args.corpus == "tiger":
        corpus = os.path.expanduser("../data/CLEANED_OLD_tiger_test.jsonl")
        #corpus = os.path.expanduser("../data/tiger_test.jsonl")
        sent_col = "Original sentence"
        gold_col = "Canonical form"
        add_space = True
    elif args.corpus == "eval":
        corpus = os.path.expanduser("../data/CLEANED_evaluation_sentences.jsonl")
        #corpus = os.path.expanduser("../data/evaluation_sentences.jsonl")
        sent_col = "Sentence"
        gold_col = "Gold"
        add_space = False
    elif args.corpus == "merged":
        test_data1 = os.path.expanduser("../data/CLEANED_OLD_tiger_test.jsonl")
        test_data2 = os.path.expanduser("../data/CLEANED_OLD_tüba_test.jsonl")

        #test_data1 = os.path.expanduser("../data/tiger_test.jsonl")
        #test_data2 = os.path.expanduser("../data/tüba_test.jsonl")

        test_dataset1 = load_dataset("json", data_files=test_data1, split='train')
        test_dataset2 = load_dataset("json", data_files=test_data2, split='train')
        test_dataset2 = test_dataset2.rename_column("Treebank-Sentence", "Original sentence")
        test_dataset2 = test_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
        dataset = concatenate_datasets([test_dataset1, test_dataset2])
        corpus = "One Old Merged Test"

        dataset.to_json("MergedEvalData.jsonl")
        print("Saved dataset!")

        sent_col = "Original sentence"
        gold_col = "Canonical form"
        add_space = True 
    
    elif args.corpus == "merged50":
        test_data1 = os.path.expanduser("../data/CLEANED_OLD_tiger_test.jsonl")
        test_data2 = os.path.expanduser("../data/CLEANED_OLD_tüba_test.jsonl")

        test_dataset1 = load_dataset("json", data_files=test_data1, split='train')
        test_dataset2 = load_dataset("json", data_files=test_data2, split='train')
        test_dataset2 = test_dataset2.rename_column("Treebank-Sentence", "Original sentence")
        test_dataset2 = test_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
        
        fcr1 = test_dataset1.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
        gapping1 = test_dataset1.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
        bcr1 = test_dataset1.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
        sgf1 = test_dataset1.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")
        noCCE1 = test_dataset1.filter(lambda example: (example["SGF"] == 0 or example["SGF"] == "0") and (example["BCR"] == 0 or example["BCR"] == "0") and (example["FCR"]== 0 or example["FCR"] == "0") and (example["Gapping"] == 0 or example["Gapping"] == "0"))

        fcr2 = test_dataset2.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
        gapping2 = test_dataset2.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
        bcr2 = test_dataset2.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
        sgf2 = test_dataset2.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")
        noCCE2 = test_dataset2.filter(lambda example: (example["SGF"] == 0 or example["SGF"] == "0") and (example["BCR"] == 0 or example["BCR"] == "0") and (example["FCR"]== 0 or example["FCR"] == "0") and (example["Gapping"] == 0 or example["Gapping"] == "0"))

        final_fcr = fcr1.select(range(111))
        final_gapping = gapping1.select(range(63))
        final_bcr = bcr1.select(range(12))
        final_sgf = sgf1.select(range(29))
        final_noCCE = noCCE2.select(range(66))
        final_dataset = concatenate_datasets([final_fcr, final_gapping, final_bcr, final_sgf, final_noCCE])

        dataset = concatenate_datasets([final_dataset, fcr2, gapping2, bcr2, sgf2, noCCE1])

        corpus = "One Old Merged Test 50:50"
        sent_col = "Original sentence"
        gold_col = "Canonical form"
        add_space = True 

    else: 
        print("provide a corpus!")
    
    if args.prefix:
        prefix = args.prefix
    else: 
        prefix = ""
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    if args.corpus != "merged" and args.corpus != "merged50":
        dataset = load_dataset("json", data_files=corpus, split='train')

    fcr = dataset.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    gapping = dataset.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    bcr = dataset.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    sgf = dataset.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")

    bleu = evaluate.load("bleu")
    em_metric = evaluate.load("exact_match")

    result_file = open(prefix + "_" + args.corpus + "_evaluation_result.txt", "a")

    result_file.write("CHECKPOINT: " + checkpoint + " CORPUS: " + corpus + "\n")

    evaluate_model(result_file, bleu, em_metric, fcr, "FCR", add_space)
    evaluate_model(result_file, bleu, em_metric, gapping, "GAPPING", add_space)
    evaluate_model(result_file, bleu, em_metric, bcr, "BCR", add_space)
    evaluate_model(result_file, bleu, em_metric, sgf, "SGF", add_space)
    evaluate_model(result_file, bleu, em_metric, dataset, "ALL SENTENCES", add_space)
    
    result_file.close()
    print("DONE!")
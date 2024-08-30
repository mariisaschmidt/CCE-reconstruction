import re
import os 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import evaluate
from datasets import load_dataset

def get_predictions(ds, sc):
    inputs = ds[sc]
    inputs = [clean_sentence(p) for p in inputs]
    predictions = []
    for input in inputs:
        evaluation_input = (tokenizer.encode(input, return_tensors="pt"))
        evaluation_output = model.generate(evaluation_input, max_new_tokens=200)
        decoded = tokenizer.decode(evaluation_output[0])
        decoded = decoded[6:len(decoded)-4]
        decoded = decoded.replace("<unk>", "Oe")
        predictions.append(decoded)
    return predictions

def clean_sentence(sentence):
    suffix = r'(\$_\S*)'
    sentence = re.sub(suffix, '', sentence)
    sentence = sentence.replace("$$", "")
    sentence = sentence.replace("[", "")
    sentence = sentence.replace("]", "")
    suffix2 = r'_[^\s]*'
    sentence = re.sub(suffix2, '', sentence)
    # remove spaces before punctuation
    pattern = r'\s+([.,;?!:])'
    sentence = re.sub(pattern, r'\1', sentence)
    # remove weird ``
    sentence = re.sub(r'``', '"', sentence)
    sentence = re.sub(r"''", '"', sentence)
    # replace "umlaute"
    sentence = sentence.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    sentence = sentence.replace("\/", "")
    return sentence

def evaluate_model(file, bleu, exmatch, dataset, name):
    predictions = get_predictions(dataset, sent_col)
    #predictions = [clean_sentence(p) for p in predictions]
    golds = dataset[gold_col]
    golds = [clean_sentence(s) for s in golds]

    file.write("======================" + name + "============================== \n")
    for j in range(0,2): # define multiple evaluation loops
        for i in range(0, len(predictions)):
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

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        print("You need to specify the path to the model checkpoint you want to load!")
        checkpoint = " "
    
    if args.corpus == "tuba":
        corpus = os.path.expanduser("~/data/CLEANED_tüba_test.jsonl")
        sent_col = "Treebank-Sentence"
        gold_col = "Reconstructed-Sentence"
    elif args.corpus == "tiger":
        corpus = os.path.expanduser("~/data/CLEANED_tiger_test.jsonl")
        sent_col = "Original sentence"
        gold_col = "Canonical form"
    elif args.corpus == "eval":
        corpus = os.path.expanduser("~/data/CLEANED_evaluation_sentences.jsonl")
        sent_col = "Sentence"
        gold_col = "Gold"
    else: 
        print("provide a corpus!")
    
    if args.prefix:
        prefix = args.prefix
    else: 
        prefix = ""
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    dataset = load_dataset("json", data_files=corpus, split='train')

    fcr = dataset.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    gapping = dataset.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    bcr = dataset.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    sgf = dataset.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")

    bleu = evaluate.load("bleu")
    em_metric = evaluate.load("exact_match")

    result_file = open(prefix + "_" + args.corpus + "_evaluation_result.txt", "a")

    result_file.write("CHECKPOINT: " + checkpoint + " CORPUS: " + corpus + "\n")

    evaluate_model(result_file, bleu, em_metric, fcr, "FCR")
    evaluate_model(result_file, bleu, em_metric, gapping, "GAPPING")
    evaluate_model(result_file, bleu, em_metric, bcr, "BCR")
    evaluate_model(result_file, bleu, em_metric, sgf, "SGF")
    evaluate_model(result_file, bleu, em_metric, dataset, "ALL SENTENCES")
    
    result_file.close()
    print("DONE!")
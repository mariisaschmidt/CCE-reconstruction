import re
import os 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import evaluate
import datasets

def get_predictions(inputs):
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

def add_one_space(sentence):
    return sentence + " "

def remove_brackets_and_suffix(sentence):
    # converts [word cce-type] to word
    cleaned_sentence = re.sub(r'\[([^\s\]]+)\s[^\]]+\]', r'\1', sentence)
    return cleaned_sentence

def evaluate_model(file, bleu, exmatch, dataset, name, add_space):
    inputs = dataset[sent_col]
    predictions = get_predictions(inputs)
    golds = dataset[gold_col]
    if add_space:
        for i in range(0, len(predictions)):
            if golds[i].endswith(" "):
                predictions[i] = add_one_space(predictions[i])

    file.write("======================" + name + "============================== \n")
    for j in range(0,2): # define multiple evaluation loops
        for i in range(0, len(predictions)):
            predictions[i] = remove_brackets_and_suffix(predictions[i])
            golds[i] = remove_brackets_and_suffix(golds[i])
            if j == 0:
                file.write("====================== PRED VS GOLD ============================== \n")
                file.write("pred: " + predictions[i] + "\n")
                file.write("gold: " + golds[i] + "\n")
                file.write("inpt: " + inputs[i] + "\n")
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
    parser.add_argument("--prefix", type=str)
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        print("You need to specify the path to the model checkpoint you want to load!")
        checkpoint = " "
    
    if args.prefix:
        prefix = args.prefix
    else: 
        prefix = ""
    
    add_space = True
    gold_col = "Target"
    sent_col = "Masked"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    dataset = datasets.load_from_disk(os.path.expanduser("../data/ECBAE_EvalDataset"))

    fcr = dataset.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    gapping = dataset.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    bcr = dataset.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    sgf = dataset.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")

    bleu = evaluate.load("bleu")
    em_metric = evaluate.load("exact_match")

    result_file = open(prefix + "_evaluation_result.txt", "a")

    result_file.write("CHECKPOINT: " + checkpoint + " CORPUS: " + "Masked Eval Data" + "\n")

    evaluate_model(result_file, bleu, em_metric, fcr, "FCR", add_space)
    evaluate_model(result_file, bleu, em_metric, gapping, "GAPPING", add_space)
    evaluate_model(result_file, bleu, em_metric, bcr, "BCR", add_space)
    evaluate_model(result_file, bleu, em_metric, sgf, "SGF", add_space)
    evaluate_model(result_file, bleu, em_metric, dataset, "ALL SENTENCES", add_space)
    
    result_file.close()
    print("DONE!")
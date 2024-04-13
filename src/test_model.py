import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import evaluate
from datasets import load_dataset

def get_predictions(ds, sc):
    inputs = ds[sc]
    predictions = []
    for input in inputs:
        evaluation_input = (tokenizer.encode(input, return_tensors="pt"))
        evaluation_output = model.generate(evaluation_input, max_new_tokens=200)
        decoded = tokenizer.decode(evaluation_output[0])
        decoded = decoded[6:len(decoded)-4]
        predictions.append(decoded)
    return predictions

def calculate_distance(s, p):
    m = len(s)
    n = len(p)
    d = [[0] * (n + 1) for i in range(m + 1)]  

    for i in range(1, m + 1):
        d[i][0] = i

    for j in range(1, n + 1):
        d[0][j] = j
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == p[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,      # deletion
                          d[i][j - 1] + 1,      # insertion
                          d[i - 1][j - 1] + cost) # substitution   

    return d[m][n]

def remove_suffix(sentence):
    suffix = r'(_\S*)'
    sentence = re.sub(suffix, '', sentence)
    sentence = sentence.replace("$$", "")
    sentence = sentence.replace("[", "")
    sentence = sentence.replace("]", "")
    return sentence

def evaluate_model(file, bleu, dataset, name):
    predictions = get_predictions(dataset, sent_col)
    golds = dataset[gold_col]
    goldsWithoutSuffix = [remove_suffix(s) for s in golds]
    avg_dist = 0
    avg_dist_wos = 0

    file.write("======================" + name + "============================== \n")
    for i in range(0, len(predictions)):
        d = calculate_distance(predictions[i], golds[i])
        d_wos = calculate_distance(predictions[i], goldsWithoutSuffix[i])
        ratio = d_wos / len(goldsWithoutSuffix[i])
        #r = "Distance (No Suffix): " + str(d_wos) + "\t Distance (W/ Suffix): " + str(d) + "\t Length of Sentence: " + str(len(goldsWithoutSuffix[i])) + "\t Ratio (dist/len WoS): " + str(ratio) + "\n" 
        #file.write(r)
        #r2 = "Sentence: " + predictions[i] + "\t Gold: " + golds[i] + "\n"
        #file.write(r2)
        avg_dist += d
        avg_dist_wos += d_wos
    if(len(predictions) != 0):
        avg_dist = avg_dist / len(predictions)
        avg_dist_wos = avg_dist_wos / len(predictions)
        file.write("Average Distance (No Suffix): " + str(avg_dist_wos) + "\t Average Distance (W/ Suffix): " + str(avg_dist) + "\n")
        score = bleu.compute(predictions=predictions, references=golds)
        file.write("Bleu Score: " + str(score) + "\n")
        file.write("\n")

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
        corpus = "tüba_test.jsonl"
        sent_col = "Treebank-Sentence"
        gold_col = "Reconstructed-Sentence"
    elif args.corpus == "tiger":
        corpus = "tiger_test.jsonl"
        sent_col = "Original sentence"
        gold_col = "Canonical form"
    elif args.corpus == "eval":
        corpus = "evaluation_sentences.jsonl"
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

    metric = evaluate.load("bleu")

    result_file = open(prefix + "_" + args.corpus + "_evaluation_result.txt", "a")

    evaluate_model(result_file, metric, fcr, "FCR")
    evaluate_model(result_file, metric, gapping, "GAPPING")
    evaluate_model(result_file, metric, bcr, "BCR")
    evaluate_model(result_file, metric, sgf, "SGF")
    evaluate_model(result_file, metric, dataset, "ALL SENTENCES")
    
    result_file.close()
    print("DONE!")
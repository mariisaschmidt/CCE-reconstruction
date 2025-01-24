import json
import re

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
    sentence = sentence.replace("\/", "")
    return sentence

def process_jsonl(input_file, output_file, col, gol):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            data[col] = clean_sentence(data[col])
            data[gol] = clean_sentence(data[gol])
            outfile.write(json.dumps(data) + '\n')

def add_other_golds(input_file, output_file, sentcol, goldcol, finalgoldcol): # only tiger + tüba
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if data[goldcol] != " ":
                json.dump({sentcol: data[sentcol], finalgoldcol: data[goldcol], "FCR": data["FCR"], "Gapping": data["Gapping"], "BCR": data["BCR"], "SGF": data["SGF"]}, outfile)
                outfile.write("\n")

print("Getting other gold standards!")
print("Tiger Train")
input_file = '/home/marisa/data/OLD_tiger_train.jsonl'
output_file = '/home/marisa/data/ALL_GOLDS_OLD_tiger_train.jsonl'
#input_file = '/home/marisa/data/tiger_train.jsonl'
#output_file = '/home/marisa/data/ALL_GOLDS_tiger_train.jsonl'
add_other_golds(input_file, output_file, "Original sentence", "gold2 (LCO)", "Canonical form")
# add_other_golds(input_file, output_file, "Original sentence", "gold2", "Canonical form")
# add_other_golds(input_file, output_file, "Original sentence", "gold3", "Canonical form")
add_other_golds(input_file, output_file, "Original sentence", "Canonical form", "Canonical form")
print("Tiger Test")
input_file = '/home/marisa/data/OLD_tiger_test.jsonl'
output_file = '/home/marisa/data/ALL_GOLDS_OLD_tiger_test.jsonl'
#input_file = '/home/marisa/data/tiger_test.jsonl'
#output_file = '/home/marisa/data/ALL_GOLDS_tiger_test.jsonl'
add_other_golds(input_file, output_file, "Original sentence", "gold2 (LCO)", "Canonical form")
# add_other_golds(input_file, output_file, "Original sentence", "gold2", "Canonical form")
# add_other_golds(input_file, output_file, "Original sentence", "gold3", "Canonical form")
add_other_golds(input_file, output_file, "Original sentence", "Canonical form", "Canonical form")

print("TüBa Train")
input_file = '/home/marisa/data/OLD_tüba_train.jsonl'
output_file = '/home/marisa/data/ALL_GOLDS_OLD_tüba_train.jsonl'
#input_file = '/home/marisa/data/tüba_train.jsonl'
#output_file = '/home/marisa/data/ALL_GOLDS_tüba_train.jsonl'
add_other_golds(input_file, output_file, "Treebank-Sentence", "Gold_1", "Reconstructed-Sentence")
add_other_golds(input_file, output_file, "Treebank-Sentence", "Gold_2", "Reconstructed-Sentence")
add_other_golds(input_file, output_file, "Treebank-Sentence", "Gold_3", "Reconstructed-Sentence")
add_other_golds(input_file, output_file, "Treebank-Sentence", "Reconstructed-Sentence", "Reconstructed-Sentence")
print("TüBa Test")
input_file = '/home/marisa/data/OLD_tüba_test.jsonl'
output_file = '/home/marisa/data/ALL_GOLDS_OLD_tüba_test.jsonl'
#input_file = '/home/marisa/data/tüba_test.jsonl'
#output_file = '/home/marisa/data/ALL_GOLDS_tüba_test.jsonl'
add_other_golds(input_file, output_file, "Treebank-Sentence", "Gold_1", "Reconstructed-Sentence")
add_other_golds(input_file, output_file, "Treebank-Sentence", "Gold_2", "Reconstructed-Sentence")
add_other_golds(input_file, output_file, "Treebank-Sentence", "Gold_3", "Reconstructed-Sentence")
add_other_golds(input_file, output_file, "Treebank-Sentence", "Reconstructed-Sentence", "Reconstructed-Sentence")

print("Cleaning GC4 Data: ")
input_file = '/home/marisa/data/de_de_pairs.jsonl'
output_file = '/home/marisa/data/CLEANED_de_de_pairs.jsonl'
process_jsonl(input_file, output_file, "text", "gold_sentence")

print("Cleaning TIGER Data: ")
print("Train: ")
input_file = '/home/marisa/data/tiger_train.jsonl'
output_file = '/home/marisa/data/CLEANED_ONE_NEW_tiger_train.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_OLD_tiger_train.jsonl'
output_file = '/home/marisa/data/CLEANED_ALL_OLD_tiger_train.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_tiger_train.jsonl'
output_file = '/home/marisa/data/CLEANED_OLD_tiger_train.jsonl'
process_jsonl(input_file, output_file, "Original sentence", "Canonical form")
print("Test: ")
input_file = '/home/marisa/data/tiger_test.jsonl'
output_file = '/home/marisa/data/CLEANED_ONE_NEW_tiger_test.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_OLD_tiger_test.jsonl'
output_file = '/home/marisa/data/CLEANED_ALL_OLD_tiger_test.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_tiger_test.jsonl'
output_file = '/home/marisa/data/CLEANED_OLD_tiger_test.jsonl'
process_jsonl(input_file, output_file, "Original sentence", "Canonical form")

print("Cleaning TüBa Data: ")
print("Train: ")
input_file = '/home/marisa/data/tüba_train.jsonl'
output_file = '/home/marisa/data/CLEANED_ONE_NEW_tüba_train.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_OLD_tüba_train.jsonl'
output_file = '/home/marisa/data/CLEANED_ALL_OLD_tüba_train.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_tüba_train.jsonl'
output_file = '/home/marisa/data/CLEANED_OLD_tüba_train.jsonl'
process_jsonl(input_file, output_file, "Treebank-Sentence", "Reconstructed-Sentence")
print("Test: ")
input_file = '/home/marisa/data/tüba_test.jsonl'
output_file = '/home/marisa/data/CLEANED_ONE_NEW_tüba_test.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_OLD_tüba_test.jsonl'
output_file = '/home/marisa/data/CLEANED_ALL_OLD_tüba_test.jsonl'
input_file = '/home/marisa/data/ALL_GOLDS_tüba_test.jsonl'
output_file = '/home/marisa/data/CLEANED_OLD_tüba_test.jsonl'
process_jsonl(input_file, output_file, "Treebank-Sentence", "Reconstructed-Sentence")

print("Cleaning Eval Data: ")
input_file = '/home/marisa/data/evaluation_sentences.jsonl'
output_file = '/home/marisa/data/CLEANED_evaluation_sentences.jsonl'
process_jsonl(input_file, output_file, "Sentence", "Gold")
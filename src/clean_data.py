import json
import re

def clean_sentence(sentence):
    suffix = r'_[^\s]*'
    sentence = re.sub(suffix, '', sentence)
    # remove spaces before punctuation
    pattern = r'\s+([.,;?!:])'
    sentence = re.sub(pattern, r'\1', sentence)
    # remove weird ``
    sentence = re.sub(r'``', '"', sentence)
    sentence = re.sub(r"''", '"', sentence)
    # replace "umlaute"
    sentence = sentence.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    print(sentence)
    return sentence

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            if 'text' in data:
                data['text'] = clean_sentence(data['text'])
            outfile.write(json.dumps(data) + '\n')

print("Cleaning GC4 Data: ")
input_file = '/home/marisa/data/de_de_pairs.jsonl'
output_file = '/home/marisa/data/CLEANED_de_de_pairs.jsonl'
process_jsonl(input_file, output_file)

print("Cleaning TIGER Data: ")
print("Train: ")
input_file = '/home/marisa/data/tiger_train.jsonl'
output_file = '/home/marisa/data/CLEANED_tiger_train.jsonl'
process_jsonl(input_file, output_file)
print("Test: ")
input_file = '/home/marisa/data/tiger_test.jsonl'
output_file = '/home/marisa/data/CLEANED_tiger_test.jsonl'
process_jsonl(input_file, output_file)

print("Cleaning TüBa Data: ")
print("Train: ")
input_file = '/home/marisa/data/tüba_train.jsonl'
output_file = '/home/marisa/data/CLEANED_tüba_train.jsonl'
process_jsonl(input_file, output_file)
print("Test: ")
input_file = '/home/marisa/data/tüba_test.jsonl'
output_file = '/home/marisa/data/CLEANED_tüba_test.jsonl'
process_jsonl(input_file, output_file)

print("Cleaning Eval Data: ")
input_file = '/home/marisa/data/evaluation_sentences.jsonl'
output_file = '/home/marisa/data/CLEANED_evaluation_sentences.jsonl'
process_jsonl(input_file, output_file)
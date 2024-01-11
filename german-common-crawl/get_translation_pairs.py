import os 
from datasets import load_dataset
import nltk.data
import json

def tokenize(text):
    sents = nltk.sent_tokenize(text['raw_content'])
    for s in sents:
        s = s + '[NEXT]'
    text['raw_content'] = str(sents)
    return text

if __name__ == '__main__':
    data_files = []
    dirname = "/correct_files"
    for filename in os.listdir(dirname):
        data_files.append(dirname + "/" + filename)

    ds = load_dataset("json", data_files=data_files, split='train', streaming=True)
    ds = ds.select_columns('raw_content')

    ds_modified = ds.map(tokenize)

    sent_id = 0
    with open("de_de_pairs.jsonl", "a") as file:
        for tokenized_sentence in ds_modified:
            splitted = tokenized_sentence['raw_content'].split("',")
            for i in range(len(splitted)):
                splitted[i] = splitted[i][2:(len(splitted[i])-1)]
                jsn = {"id": sent_id, "text": splitted[i], "gold_sentence": splitted[i]}
                jsn = json.dumps(jsn)
                file.write(jsn + '\n')
                sent_id += 1
                if sent_id % 100000 == 0:
                    print(sent_id)
            if sent_id > 1000000:
                break 
import pandas as pd
import numpy as np
import json
import ast 
import os

def dump_json(data):
    return json.dumps(data)

def format_json(data):
    return ast.literal_eval(data)

if __name__ == '__main__':
    df = pd.read_csv("20230426_CCE_BLEU.csv", sep=";")
    df = df.fillna(np.nan).replace([np.nan], [" "])
    df = df.astype(str)

    # create json string for each line in df
    for i in df.index:
        filename = "tmp_json/_" + str(i) + "_.json"
        df.iloc[i].to_json(filename)

    dirname = "tmp_json"
    i = 0
    for filename in os.listdir(dirname):
        with open("tmp_json/" + filename, "r") as file:
            lines = list(file)
        
        #split_indx = (len(lines)//100) * 80

        train_file = open("tüba_train.jsonl", "a")
        test_file = open("tüba_test.jsonl", "a")
        for line in lines:
            if(i <= 1442):
                train_file.write(dump_json(format_json(line)) + '\n')
                i += 1
            else:
                test_file.write(dump_json(format_json(line)) + '\n')
                i += 1
        print("Done with: " + filename)
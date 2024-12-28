import pandas as pd
import numpy as np
import json
import ast 
import os

def dump_json(data):
    return json.dumps(data)

def format_json(data):
    return ast.literal_eval(data)
# erweitert hat 13433 Sätze
if __name__ == '__main__':
    df = pd.read_csv("/Users/marisa/data/ErweiterterTIGER/TIGER_erweitert.csv", sep=";") # 0809-TIGER-canonical-forms.csv
    df = df.fillna(np.nan).replace([np.nan], [" "])
    df = df.astype(str)

    # create json string for each line in df
    for i in df.index:
        filename = "tmp_json2/_" + str(i) + "_.json"
        df.iloc[i].to_json(filename)

    dirname = "tmp_json2"
    i = 0
    for filename in os.listdir(dirname):
        with open("tmp_json2/" + filename, "r") as file:
            lines = list(file)
        
        #split_indx = (len(lines)//100) * 80

        train_file = open("erweitert_tiger_train.jsonl", "a")
        #test_file = open("erweitert_tiger_test.jsonl", "a")
        for line in lines:
            train_file.write(dump_json(format_json(line)) + '\n') # remove if if is used
            # if(i <= 10746): #5847 #2950
            #     train_file.write(dump_json(format_json(line)) + '\n')
            #     i += 1
            # else:
            #     test_file.write(dump_json(format_json(line)) + '\n')
            #     i += 1
        print("Done with: " + filename)
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
    for filename in os.listdir(dirname):
        with open("tmp_json/" + filename, "r") as file:
            lines = list(file)

        with open("cce_bleu.jsonl", "a") as correct_file:
                for line in lines:
                    correct_file.write(dump_json(format_json(line)) + '\n')
        print("Done with: " + filename)
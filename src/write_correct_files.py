import json 
import pandas as pd 
import ast
import os 

def dump_json(data):
    return json.dumps(data)

def format_json(data):
    return ast.literal_eval(line)

if __name__ == '__main__':
    dirname = "base_files"
    for filename in os.listdir(dirname):
        with open(dirname + "/" + filename, "r") as file:
            lines = list(file)

        with open("correct_files/" + filename, "w") as correct_file:
            for line in lines:
                correct_file.write(dump_json(format_json(line)) + '\n')
        print("Done with: " + filename)
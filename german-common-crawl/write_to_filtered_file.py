#!/usr/bin/env python3
import tarfile
from ast import literal_eval
from tqdm import tqdm
import sys

def tar_file_to_string(file_name):
    with tarfile.open(file_name, "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            data = f.readline()
            data = data.decode("utf-8")
            data = data.split("{'url'")
            data = [("{'url'" + item) for item in data]
            data = data[1:]
    return data

if __name__ == '__main__':
    file_name = sys.argv[1]
    new_file_name = ".".join(file_name.split(".")[:1] + ["jsonl"])
    a = tar_file_to_string(file_name)
    b = []
    for item in tqdm(a):
        try:
            if literal_eval(item)['language_score'] > 0.98:
                b.append(item)
        except:
            None
    with open(new_file_name, 'wt') as file_new:
        for part in b:
            file_new.write(part + '\n')
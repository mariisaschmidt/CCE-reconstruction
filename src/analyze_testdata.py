import os 
from datasets import load_dataset, concatenate_datasets
import nltk
import numpy as np 

def get_sentence_lengths(text):
    sentences = nltk.sent_tokenize(text)  # Sätze aufteilen
    word_counts = [len(sentence.split()) for sentence in sentences]  # Wörter pro Satz zählen
    return word_counts

def get_avg_length(dataset, text_column):
    all_lengths = [length for example in dataset[text_column] for length in get_sentence_lengths(example)]
    average_length = np.mean(all_lengths)
    print(f"Durchschnittliche Anzahl Wörter pro Satz: {average_length:.4f} Wörter. \n")

if __name__ == '__main__':
    train_data = os.path.expanduser("~/data/CLEANED_erweitert_tiger_train.jsonl")
    dataset = load_dataset("json", data_files=train_data, split='train')
    
    # test_data1 = os.path.expanduser("~/data/CLEANED_OLD_tiger_test.jsonl")
    # test_data2 = os.path.expanduser("~/data/CLEANED_OLD_tüba_test.jsonl")

    # test_dataset1 = load_dataset("json", data_files=test_data1, split='train')
    # test_dataset2 = load_dataset("json", data_files=test_data2, split='train')
    # test_dataset2 = test_dataset2.rename_column("Treebank-Sentence", "Original sentence")
    # test_dataset2 = test_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
    # dataset = concatenate_datasets([test_dataset1, test_dataset2])

    # fcr1 = test_dataset1.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    # gapping1 = test_dataset1.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    # bcr1 = test_dataset1.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    # sgf1 = test_dataset1.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")
    # noCCE1 = test_dataset1.filter(lambda example: (example["SGF"] == 0 or example["SGF"] == "0") and (example["BCR"] == 0 or example["BCR"] == "0") and (example["FCR"]== 0 or example["FCR"] == "0") and (example["Gapping"] == 0 or example["Gapping"] == "0"))

    # fcr2 = test_dataset2.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    # gapping2 = test_dataset2.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    # bcr2 = test_dataset2.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    # sgf2 = test_dataset2.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")
    # noCCE2 = test_dataset2.filter(lambda example: (example["SGF"] == 0 or example["SGF"] == "0") and (example["BCR"] == 0 or example["BCR"] == "0") and (example["FCR"]== 0 or example["FCR"] == "0") and (example["Gapping"] == 0 or example["Gapping"] == "0"))

    # fcr = dataset.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    # gapping = dataset.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    # bcr = dataset.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    # sgf = dataset.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")
    # noCCE = dataset.filter(lambda example: (example["SGF"] == 0 or example["SGF"] == "0") and (example["BCR"] == 0 or example["BCR"] == "0") and (example["FCR"]== 0 or example["FCR"] == "0") and (example["Gapping"] == 0 or example["Gapping"] == "0"))

    print("Hier ein paar Info's über die Testdaten: ")
    # print("TIGER - Gesamt")
    # print(test_dataset1)
    # print("TIGER - FCR")   
    # print(fcr1)
    # print("TIGER - GAPPING") 
    # print(gapping1)
    # print("TIGER - BCR") 
    # print(bcr1)
    # print("TIGER - SGF")  
    # print(sgf1)
    # print("TIGER - NoCCE")  
    # print(noCCE1)
    # print("======================================== \n")
    # print("TüBa - Gesamt")
    # print(test_dataset2)
    # print("TüBa - FCR")   
    # print(fcr2)
    # print("TüBa - GAPPING") 
    # print(gapping2)
    # print("TüBa - BCR") 
    # print(bcr2)
    # print("TüBa - SGF")  
    # print(sgf2)
    # print("TüBa - NoCCE")  
    # print(noCCE2)
    # print("======================================== \n")
    print("MERGED - Gesamt")
    print(dataset)
    # print("MERGED - FCR")   
    # print(fcr)
    # print("MERGED - GAPPING") 
    # print(gapping)
    # print("MERGED - BCR") 
    # print(bcr)
    # print("MERGED - SGF")  
    # print(sgf)
    # print("MERGED - NoCCE")  
    # print(noCCE)

    print("SATZLÄNGE EVAL")
    # print("Tiger: ")
    # get_avg_length(test_dataset1, "Canonical form")
    # print("Tüba: ")
    # get_avg_length(test_dataset2, "Canonical form")
    print("Merged: ")
    get_avg_length(dataset, "Canonical form")
    # print("FCR: ")
    # get_avg_length(fcr, "Canonical form")
    # print("BCR: ")
    # get_avg_length(bcr, "Canonical form")
    # print("Gapping: ")
    # get_avg_length(gapping, "Canonical form")
    # print("SGF: ")
    # get_avg_length(sgf, "Canonical form")
    # print("NoCCE: ")
    # get_avg_length(noCCE, "Canonical form")

    # print("Creating a 50:50 Dataset now:")
    # final_fcr = fcr1.select(range(111))
    # final_gapping = gapping1.select(range(63))
    # final_bcr = bcr1.select(range(12))
    # final_sgf = sgf1.select(range(29))
    # final_noCCE = noCCE2.select(range(66))
    # final_dataset = concatenate_datasets([final_fcr, final_gapping, final_bcr, final_sgf, final_noCCE])

    # fifty_fifty_dataset = concatenate_datasets([final_dataset, fcr2, gapping2, bcr2, sgf2, noCCE1])

    # print("SATZLÄNGE")
    # get_avg_length(fifty_fifty_dataset, "Canonical form")

    # print("======================================== \n")
    # print("50:50 - Gesamt")
    # print(fifty_fifty_dataset)


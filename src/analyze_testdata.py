import os 
from datasets import load_dataset, concatenate_datasets

if __name__ == '__main__':
    test_data1 = os.path.expanduser("~/data/CLEANED_OLD_tiger_test.jsonl")
    test_data2 = os.path.expanduser("~/data/CLEANED_OLD_tüba_test.jsonl")

    test_dataset1 = load_dataset("json", data_files=test_data1, split='train')
    test_dataset2 = load_dataset("json", data_files=test_data2, split='train')
    test_dataset2 = test_dataset2.rename_column("Treebank-Sentence", "Original sentence")
    test_dataset2 = test_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
    dataset = concatenate_datasets([test_dataset1, test_dataset2])

    fcr1 = test_dataset1.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    gapping1 = test_dataset1.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    bcr1 = test_dataset1.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    sgf1 = test_dataset1.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")

    fcr2 = test_dataset2.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    gapping2 = test_dataset2.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    bcr2 = test_dataset2.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    sgf2 = test_dataset2.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")

    fcr = dataset.filter(lambda example: example["FCR"] == 1 or example["FCR"] == "1")
    gapping = dataset.filter(lambda example: example["Gapping"] == 1 or example["Gapping"] == "1")
    bcr = dataset.filter(lambda example: example["BCR"] == 1 or example["BCR"] == "1")
    sgf = dataset.filter(lambda example: example["SGF"] == 1 or example["SGF"] == "1")

    print("Hier ein paar Info's über die Testdaten: ")
    print("TIGER - Gesamt")
    print(test_dataset1)
    print("TIGER - FCR")   
    print(fcr1)
    print("TIGER - GAPPING") 
    print(gapping1)
    print("TIGER - BCR") 
    print(bcr1)
    print("TIGER - SGF")  
    print(sgf1)
    print("======================================== \n")
    print("TüBa - Gesamt")
    print(test_dataset2)
    print("TüBa - FCR")   
    print(fcr2)
    print("TüBa - GAPPING") 
    print(gapping2)
    print("TüBa - BCR") 
    print(bcr2)
    print("TüBa - SGF")  
    print(sgf2)
    print("======================================== \n")
    print("MERGED - Gesamt")
    print(dataset)
    print("TüBa - FCR")   
    print(fcr)
    print("TüBa - GAPPING") 
    print(gapping)
    print("TüBa - BCR") 
    print(bcr)
    print("TüBa - SGF")  
    print(sgf)
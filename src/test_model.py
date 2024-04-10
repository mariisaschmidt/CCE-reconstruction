from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import evaluate
from datasets import load_dataset

def get_prediction(examples):
    inputs = examples["Sentence"]
    evaluation_input = (tokenizer.encode(inputs, return_tensors="pt"))
    evaluation_output = model.generate(evaluation_input, max_new_tokens=200)
    decoded = tokenizer.decode(evaluation_output[0])
    decoded = decoded[6:len(decoded)-4]
    return decoded

def calculate_distance():
    pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--sentence", type=str)
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        print("You need to specify the path to the model checkpoint you want to load!")
        checkpoint = " "
    if args.sentence:
        sentence = args.sentence
    else:
        sentence = "Die Sonne ist der Stern, der der Erde am nächsten ist und das Zentrum des Sonnensystems bildet."
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    dataset = load_dataset("json", data_files="/Users/marisa/CCE-reconstruction/evaluation_sentences.jsonl", split='train')
    predictions = dataset.map(get_prediction)
    print(predictions)

        
    #metric = evaluate.load("bleu")
    #results = metric.compute(predictions=predictions, references=evaluation_sentences)
    #print(results)
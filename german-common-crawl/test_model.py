from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import evaluate

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

    evaluation_sentences = ["Hallo, wie geht es dir?", 
                            "Die vorhandene Handlungsbereitschaft wurde in Entscheidungen und Arbeitspakete umgesetzt", 
                            "Dabei vermengen sich die Konfliktlagen und die Interessen der Akteure",
                            "In den Wahlkampf wird sein Kandidat nicht eingreifen , er bleibt bis September zu Hause .",
                            "In der Theorie schafft es der Wettbewerb , durch den Konkurrenzgedanken das Streben nach Monopolen und Macht systematisch auszuschalten ."]
    evaluation_sentences.append(sentence)
    
    predictions = []
    for s in evaluation_sentences:
        evaluation_input = (tokenizer.encode(s, return_tensors="pt"))
        evaluation_output = model.generate(evaluation_input, max_new_tokens=200)
        decoded = tokenizer.decode(evaluation_output[0])
        decoded = decoded[6:len(decoded)-4]
        print(s + " -> " + decoded)
        predictions.append(decoded)
        
    metric = evaluate.load("bleu")
    results = metric.compute(predictions=predictions, references=evaluation_sentences)
    print(results)
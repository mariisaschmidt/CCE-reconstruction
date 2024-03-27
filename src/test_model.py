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

    evaluation_sentences = ["Anita kennt jede Zeile , jedes Wort .", # Anita kennt jede Zeile , Anita_f kennt_f jedes Wort . 
                            "Er war etwa 23 Jahre alt und sah wie ein frecher Junge aus .", # Er war etwa 23 Jahre alt und er_f sah wie ein frecher Junge aus . 
                            "Ich glaube , es wird genausoviel Trennung von Parlament und Bürger geben .", #;Ich glaube genausoviel_bcr Trennung_bcr von_bcr Parlament_bcr und_bcr Bürger_bcr geben_g , es wird genausoviel Trennung von Parlament und Bürger geben .
                            "Dabei vermengen sich die Konfliktlagen und die Interessen der Akteure",
                            "In den Wahlkampf wird sein Kandidat nicht eingreifen , er bleibt bis September zu Hause .",
                            "In der Theorie schafft es der Wettbewerb , durch den Konkurrenzgedanken das Streben nach Monopolen und Macht systematisch auszuschalten .",
                            "Da die Kapazität des Stalles nur für 50 Pferde ausreicht , bleiben die Tiere mitunter auf dem Lkw und kommen am folgenden Tag unversorgt auf den nächsten Transporter oder - jetzt fast immer - auf den Zug . ", #Da die Kapazität des Stalles nur für 50 Pferde ausreicht , bleiben die Tiere mitunter auf dem Lkw und die_s Tiere_s kommen am folgenden Tag unversorgt auf den nächsten Transporter oder - jetzt fast immer - auf den Zug . 
                            "Am Freitag wurde in rund 15 Universitäten und Hochschulen protestiert , erstmals auch an Fakultäten in Paris . ", #Am Freitag wurde in rund 15 Universitäten und Hochschulen protestiert , erstmals wurde_g am_g Freitag_g auch an Fakultäten in Paris protestiert_g . 
                            #"",
    ]
    evaluation_sentences.append(sentence)
    
    predictions = []
    for s in evaluation_sentences:
        evaluation_input = (tokenizer.encode(s, return_tensors="pt"))
        evaluation_output = model.generate(evaluation_input, max_new_tokens=200)
        decoded = tokenizer.decode(evaluation_output[0])
        decoded = decoded[6:len(decoded)-4]
        print(s + "\n -> " + decoded)
        predictions.append(decoded)
        
    metric = evaluate.load("bleu")
    results = metric.compute(predictions=predictions, references=evaluation_sentences)
    print(results)
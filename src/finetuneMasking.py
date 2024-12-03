import datasets
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch 
import evaluate
import argparse
import numpy as np
import gc 
from accelerate import load_checkpoint_and_dispatch

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result = {"bleu": result["bleu"]}

    bleu_scores = []
    for pred, label in zip(preds, labels):
        pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
        label = np.where(label != -100, label, tokenizer.pad_token_id)

        decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label, skip_special_tokens=True)

        # Berechnung der Metrik für einzelne Prädiktion
        bleu_score = metric.compute(predictions=[decoded_pred], references=[[decoded_label]])["bleu"]
        bleu_scores.append(bleu_score)

    # Durchschnitt über alle BLEU-Scores
    result = {"bleu": round(np.mean(bleu_scores), 4)}

    # Speicher freigeben
    del preds, labels, decoded_preds, decoded_labels
    gc.collect()

    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Tokenisierung
def tokenize_function(example):
    inputs = tokenizer(example["Masked"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(example["Target"], padding="max_length", truncation=True, max_length=128)
    
    inputs["labels"] = targets["input_ids"]
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pretrained_model", type=str)
    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "MaskedT5"
    if args.pretrained_model:
        checkpoint = args.pretrained_model
    else:
        print("No checkpoint provided, defaulting to t5-small.")
        checkpoint = "t5-small"

    print("CHECK:", checkpoint)

    dataset = datasets.load_from_disk(os.path.expanduser("~/data/MaskedTrainTestDataset"))
    batchsize = 1
    epochs = 1

    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(tokenized_dataset)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#    torch.backends.cuda.matmul.allow_tf32 = True
#    torch.cuda.set_per_process_memory_fraction(0.9)

    model = T5ForConditionalGeneration.from_pretrained(checkpoint) #.to(device)
    model = load_checkpoint_and_dispatch(
        model,
        "t5-large",
        device_map="auto",  # Automatische Verteilung der Modellteile
        offload_folder="offload",  # Speicherort für Offloading
    )
    print(model.hf_device_map)
    
    metric = evaluate.load("bleu")
    log_dir = os.path.expanduser("~/models/" + model_name + "/logs")

    training_args = TrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=0,
        save_strategy="epoch",
        logging_dir=log_dir,
        push_to_hub=False,
#        optim="adafactor",
        fp16=True,
#        gradient_accumulation_steps=2,
#        gradient_checkpointing=True,
        log_level="debug",
    )

    # Trainer einrichten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # Kann auch ein separater Validierungsdatensatz sein
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Training starten
    print("Train Model: ")
    trainer.train()
    print("Saving Model ..")
    trainer.save_model(output_dir=os.path.expanduser("~/models/" + model_name))

# def predict(model, tokenizer, input_text):
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
#     outputs = model.generate(**inputs, num_beams=5, num_return_sequences=5)
#     return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# test_input = "Meine Schwester fährt ein rotes Auto und <extra_id_0> wohnt in Bayern."
# preds = predict(model, tokenizer, test_input)
# for pred in preds:
#     print(pred)
import datasets
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch 
import evaluate
import argparse
import numpy as np
import gc 

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    with torch.no_grad():
        preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [np.where(pred != -100, pred, tokenizer.pad_token_id) for pred in decoded_preds]
        decoded_labels = [np.where(label != -100, label, tokenizer.pad_token_id) for label in decoded_labels]

        bleu_score = metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])["bleu"]

    del preds, labels, decoded_preds, decoded_labels
    gc.collect()

    result = {"bleu": round(bleu_score, 4)}

    return result

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

    # Datensatz laden
    # dataset = datasets.load_from_disk(os.path.expanduser("../data/FairMasked+CCE_TrainTestDataset"))
    dataset = datasets.load_from_disk(os.path.expanduser("../data/ECBAEMasked_TrainTestDataset"))

    batchsize = 4
    epochs = 5

    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(tokenized_dataset)

    # Device auswählen
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Modell und Metrik laden
    model = T5ForConditionalGeneration.from_pretrained(checkpoint) 
    metric = evaluate.load("bleu")
    log_dir = os.path.expanduser("../models/" + model_name + "/logs")

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
        fp16=True,
        log_level="debug",
    )

    # Trainer einrichten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # Kann auch ein separater Validierungsdatensatz sein
        tokenizer=tokenizer,
    )

    # Training starten
    print("Train Model: ")
    trainer.train()
    print("Saving Model ..")
    trainer.save_model(output_dir=os.path.expanduser("../models/" + model_name))
# the all-in-one script to finetune a huggingface model 
# run like this: python3 finetune.py --dataset g4 --model_name de_de_mar10 --checkpoint t5-small

from datasets import load_dataset
from responses import target
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np 
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from datasets import DatasetDict

def preprocess_function(examples):
    inputs = prefix + examples[t]
    targets = examples[g]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True, padding='longest', return_tensors='pt')
    return model_inputs

def correct_inputs_masks_labels(examples):
    examples['input_ids'] = examples['input_ids'][0] 
    examples['attention_mask'] = examples['attention_mask'][0] 
    examples['labels'] = examples['labels'][0] 
    return examples

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["bleu"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pretrained_model", type=str)
    args = parser.parse_args()

    if args.dataset:
        dataset_name = args.dataset
    else:
        print("You need to specify a dataset! Options are: tüba, tiger, g4")
        dataset_name = ""
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "de_de_llm"
    if args.pretrained_model:
        checkpoint = args.pretrained_model
    else:
        if dataset_name == "g4":
            checkpoint = "t5-small"
        else: 
            print("You need to provide a german llm for finetuning with ellipsis data!")
    
    if dataset_name != "":
        if dataset_name == "tüba":
            train_data = "tüba_train.jsonl"
            test_data = "tüba_test.jsonl"
            train_dataset = load_dataset("json", data_files=train_data, split='train')
            print("Got train data")
            test_dataset = load_dataset("json", data_files=train_data, split='train')
            print("Got test data")
            dataset = DatasetDict({"train": train_dataset,
                                        "test": test_dataset
                                        })
            t = "Treebank-Sentence"
            g = "Reconstructed-Sentence"
            prefix = "reconstruct the ellipsis in this sentence: "
            batchsize = 4
            epochs = 5
        if dataset_name == "tiger":
            train_data = "tiger_train.jsonl"
            test_data = "tiger_test.jsonl"
            train_dataset = load_dataset("json", data_files=train_data, split='train')
            print("Got train data")
            test_dataset = load_dataset("json", data_files=train_data, split='train')
            print("Got test data")
            dataset = DatasetDict({"train": train_dataset,
                                        "test": test_dataset
                                        })
            t = "Original sentence"
            g = "Canonical form"
            batchsize = 4
            prefix = "reconstruct the ellipsis in this sentence: "
            epochs = 10
        if dataset_name == "g4":
            data = "de_de_pairs.jsonl"
            dataset = load_dataset("json", data_files=data, split='train')
            prefix = "translate German to German: "
            t = "text"
            g = "gold_sentence"
            batchsize = 16
            epochs = 2
        
        print("Loaded Dataset!")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        source_lang = "de"
        target_lang = "de"

        dataset = dataset.filter(lambda example: len(example[t]) >= 20)
        dataset = dataset.filter(lambda example: len(example[g]) >= 20)

        print("Preprocess Data: ")
        tokenized_dataset = dataset.map(preprocess_function, batched=False)

        print("Correct the outputs of preprocess: ")
        tokenized_dataset = tokenized_dataset.map(correct_inputs_masks_labels, batched=False)

        if dataset_name == "gcc":
            print("Create Train-Test-Split: ")
            tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)   

        metric = evaluate.load("bleu")

        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

        training_args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batchsize,
        per_device_eval_batch_size=batchsize,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True, # set true when cuda available
        push_to_hub=False,
        generation_max_length=256,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("Train Model: ")
        trainer.train()
        print("Saving Model ..")
        trainer.save_model()
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
    inputs = prefix + examples['Original sentence']
    targets = examples["Canonical form"]
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
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pretrained_model", type=str)
    args = parser.parse_args()

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "de_de_finetuned4ellipsis"
    if args.pretrained_model:
        checkpoint = args.pretrained_model
    else:
        checkpoint = "de_de_llm"

    train_data = "tiger_train.jsonl"
    test_data = "tiger_test.jsonl"
    train_dataset = load_dataset("json", data_files=train_data, split='train')
    print("Got train data")
    test_dataset = load_dataset("json", data_files=train_data, split='train')
    print("Got test data")
    de_de_dataset = DatasetDict({"train": train_dataset,
                                 "test": test_dataset
                                })
    print("Loaded Dataset!")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    source_lang = "de"
    target_lang = "de"
    prefix = "reconstruct the ellipsis in this sentence: "

    de_de_dataset = de_de_dataset.filter(lambda example: len(example['Original sentence']) >= 5)
    de_de_dataset = de_de_dataset.filter(lambda example: len(example['Canonical form']) >= 5)

    print("Preprocess Data: ")
    tokenized_dataset = de_de_dataset.map(preprocess_function, batched=False)

    print("Correct the outputs of preprocess: ")
    tokenized_dataset = tokenized_dataset.map(correct_inputs_masks_labels, batched=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)   

    metric = evaluate.load("bleu")

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    training_args = Seq2SeqTrainingArguments(
    output_dir=model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True, # set true when cuda available
    push_to_hub=False,
    generation_max_length=256
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



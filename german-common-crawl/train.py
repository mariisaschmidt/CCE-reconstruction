from datasets import load_dataset
from responses import target
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np 
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def preprocess_function(examples):
    inputs = prefix + examples['text']
    targets = examples["gold_sentence"]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding='longest', return_tensors='pt')
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
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == '__main__':
    data = "/Users/marisa/clausal-coordinate-ellipsis/german-common-crawl/de_de_pairs.jsonl"
    de_de_dataset = load_dataset("json", data_files=data, split='train')
    print("Loaded Dataset!")

    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    source_lang = "de"
    target_lang = "de"
    prefix = "translate German to German: "

    de_de_dataset = de_de_dataset.filter(lambda example: len(example['text']) >= 20)
    de_de_dataset = de_de_dataset.filter(lambda example: len(example['gold_sentence']) >= 20)

    print("Preprocess Data: ")
    tokenized_dataset = de_de_dataset.map(preprocess_function, batched=False)

    print("Correct the outputs of preprocess: ")
    tokenized_dataset = tokenized_dataset.map(correct_inputs_masks_labels, batched=False)

    print("Create Train-Test-Split: ")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)   

    metric = evaluate.load("sacrebleu")

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    training_args = Seq2SeqTrainingArguments(
    output_dir="de_de_110124",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True, # set true when cuda available
    push_to_hub=False,
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


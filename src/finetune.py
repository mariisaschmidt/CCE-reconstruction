# the all-in-one script to finetune a huggingface ellipsis reconstruction model 
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np 
import os 
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from datasets import DatasetDict
import optuna
import optuna.visualization as vis
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

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
    labels = [label.strip() for label in labels]

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

    # result = metric_em.compute(predictions=decoded_preds, references=decoded_labels, ignore_case=True, ignore_punctuation=True)
    # result = {"exact_match": result["exact_match"]}

    #prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    #result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# def run_optuna():
#     pruner = SuccessiveHalvingPruner()
#     sampler = TPESampler()
#     study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
#     study.optimize(lambda trial: objective(trial), n_trials=50)

#     print("Best hyperparameters:", study.best_params)

#     fig1 = vis.plot_optimization_history(study)
#     fig2 = vis.plot_param_importances(study)
#     fig3 = vis.plot_parallel_coordinate(study)

#     # Graphen speichern
#     fig1.write_image("optimization_history.png")
#     fig2.write_image("param_importances.png")
#     fig3.write_image("parallel_coordinates.png")

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def param_space(trial):
    return {
        "per_device_train_batch_size": trial.suggest_categorical            ("per_device_train_batch_size", [4, 8, 16]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 15) 
    }

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
            checkpoint = "t5-base"
        else: 
            print("You need to provide a german llm for finetuning with ellipsis data!")
    
    if dataset_name != "":
        if dataset_name == "merged":
            train_data1 = os.path.expanduser("~/data/CLEANED_tiger_train.jsonl")
            train_data2 = os.path.expanduser("~/data/CLEANED_tüba_train.jsonl")

            train_dataset1 = load_dataset("json", data_files=train_data1, split='train')
            train_dataset2 = load_dataset("json", data_files=train_data2, split='train')
            train_dataset2 = train_dataset2.rename_column("Treebank-Sentence", "Original sentence")
            train_dataset2 = train_dataset2.rename_column("Reconstructed-Sentence", "Canonical form")
            cols_to_remove1 = train_dataset1.column_names
            cols_to_remove2 = train_dataset2.column_names
            cols_to_remove2.remove("Original sentence")
            cols_to_remove2.remove("Canonical form")
            train_dataset2 = train_dataset2.remove_columns(cols_to_remove2)
            cols_to_remove1.remove("Original sentence")
            cols_to_remove1.remove("Canonical form")
            train_dataset1 = train_dataset1.remove_columns(cols_to_remove1)
            dataset = concatenate_datasets([train_dataset1, train_dataset2])
            print("Got train data")

            t = "Original sentence"
            g = "Canonical form"
            prefix = "reconstruct the ellipsis in this sentence: "

        if dataset_name == "g4":
            data = os.path.expanduser("~/data/CLEANED_de_de_pairs.jsonl")
            dataset = load_dataset("json", data_files=data, split='train')
            prefix = "translate German to German: "
            t = "text"
            g = "gold_sentence"
        
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

        print("Create Train-Test-Split: ")
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    metric = evaluate.load("bleu")
    metric_em = evaluate.load("exact_match") 

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)   
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    log_dir = os.path.expanduser("~/models/" + model_name + "/logs")

    training_args = Seq2SeqTrainingArguments(
    output_dir=model_name,
    evaluation_strategy="epoch",
    logging_dir=log_dir,
    predict_with_generate=True,
    fp16=True, # set true when cuda available
    save_strategy="no",
    push_to_hub=False,
    generation_max_length=256
    )

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Optimize Hyperparams")
    my_kwargs = {
    "sampler": optuna.samplers.TPESampler(),
    "study_name": "23Dec_study",
    "storage": "sqlite:///23Dec_BLEU_study.db",
    "load_if_exists": True
    }
    
    best = trainer.hyperparameter_search(param_space, None, 100, "maximize", "optuna", None, **my_kwargs)

    print("Train best Model: ")
    print("best params: ")
    print(best)

    best_training_args = Seq2SeqTrainingArguments(
    output_dir=model_name,
    evaluation_strategy="epoch",
    logging_dir=log_dir,
    predict_with_generate=True,
    fp16=True, # set true when cuda available
    save_strategy="no",
    push_to_hub=False,
    generation_max_length=256,
    per_device_train_batch_size=best.hyperparameters["per_device_train_batch_size"],
    learning_rate=best.hyperparameters["learning_rate"],
    weight_decay=best.hyperparameters["weight_decay"],
    num_train_epochs=best.hyperparameters["num_train_epochs"]
    )

    best_trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=best_training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    best_trainer.train()

    print("Evaluating final model:")
    metrics = best_trainer.evaluate()
    print(metrics)

    print("Saving Model ..")
    best_trainer.save_model(output_dir=os.path.expanduser("~/models/" + model_name))
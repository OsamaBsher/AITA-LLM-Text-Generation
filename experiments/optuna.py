# -*- coding: utf-8 -*-

import evaluate
import json
import pandas as pd
import re
import torch
import sys
import optuna

# Add path to bart score library
sys.path.insert(1, './BARTScore-main')

from bart_score import BARTScorer
from datasets import Dataset, DatasetDict, load_metric
from pprint import pprint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from transformers import EarlyStoppingCallback

config = None

try:
    with open("model_config.json") as f:
      config = json.load(f)  
except:
    print("Error: Could not load config file, please ensure config files is in the directory and is formatted correctly.")
    exit(1)

print("Running with the following settings: ")
pprint(config)

# The dataset
dataset_name = config["dataset"]

# Read dataset train, test and validation splits
folder_path = "./datasets/{0}/".format(dataset_name)

train_split_path = folder_path + "train_{0}".format(dataset_name) + ".csv"
val_split_path = folder_path + "val_{0}".format(dataset_name)  + ".csv"
test_split_path = folder_path + "test_{0}".format(dataset_name)  + ".csv"

train_df = pd.read_csv(train_split_path).head(2)
val_df = pd.read_csv(val_split_path).head(2)
test_df = pd.read_csv(test_split_path).head(2)

train_df = Dataset.from_pandas(train_df)
val_df = Dataset.from_pandas(val_df)
test_df = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_df,
    "test": test_df,
    "validation": val_df})

# Model initialisation 
model_type = config["model"]

AVAILABLE_MODELS = {
  "bart": "facebook/bart-base", 
  "t5": "google/flan-t5-base"
}

MODEL_TOKENIZERS = {
  "bart": BartTokenizer,
  "t5": T5Tokenizer
}

# path to model from dictionary
model_path = AVAILABLE_MODELS[model_type]

# initialise model
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# initialise tokeniser
tokenizer = MODEL_TOKENIZERS[model_type].from_pretrained(model_path)

# model tokens sequence length
token_sequence_length = config["token_sequence_length"]

def process_data(data):

  # get the posts and comments
  posts = data["text"]
  comments = data["comment"]
  
  # prefix for t5
  if model_type == "t5":
    prefix = "generate comment: "
    posts = [prefix + post for post in posts]
  else:
    posts = [post for post in posts]

  # encode the posts
  model_inputs = tokenizer(posts, max_length=token_sequence_length, padding="max_length", truncation=True)

  # encode the comments
  labels = tokenizer(comments, max_length=token_sequence_length, padding="max_length", truncation=True).input_ids
  
  # set labels
  model_inputs["labels"] = labels

  return model_inputs

# tokenise data
dataset = dataset.map(process_data, batched=True)

# create dataloaders
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=config["batch_size"])
valid_dataloader = DataLoader(dataset["validation"], batch_size=config["batch_size"])
test_dataloader = DataLoader(dataset["test"], batch_size=config["batch_size"])

# load metrics 
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bart_scorer = BARTScorer(device="cpu", checkpoint="facebook/bart-base")

# regex to extract generated comment sentiment
yta_regex = r'\byta\b|\besh\b'
yta_pattern = re.compile(yta_regex, flags=0)

nta_regex = r'\bnta\b|\bnah\b'
nta_pattern = re.compile(nta_regex, flags=0)

def find_label(post):
  if yta_pattern.search(post):
    return "yta"
  elif nta_pattern.search(post):
    return "nta"
  else:
    return None

def accuracy_score(preds, labels):
  correct = 0
  total = 0
  for index, pred in enumerate(preds):
    actual_label = find_label(labels[index])
    predicted_label = find_label(pred)
    if actual_label == predicted_label:
      correct+=1
    total+=1
  return correct/total

def compute_metrics(input):

  predictions, labels = input
  decode_predictions = list(tokenizer.batch_decode(predictions, skip_special_tokens=True))
  decode_labels = list(tokenizer.batch_decode(labels, skip_special_tokens=True))

  preds = [x for x in decode_predictions]
  ref = [[x] for x in decode_labels]

  bleu_score = bleu.compute(predictions=preds, references=ref)
  meteor_score = meteor.compute(predictions=preds, references=ref)
  bert_score = bertscore.compute(predictions=preds, references=decode_labels, model_type="distilbert-base-uncased")
  rouge_score = rouge.compute(predictions=preds, references=ref)
  bart_score = sum(bart_scorer.score(preds, decode_labels, batch_size=4)) / len(ref)

  rounded_bart = round(bart_score, 2)
  rounded_meteor = round(meteor_score["meteor"], 2)
  rounded_bleu = round(bleu_score["bleu"], 2)

  accuracy = round(accuracy_score(decode_predictions, decode_labels), 2)

  bert_precision = round(sum(bert_score["precision"]) / len(bert_score["precision"]), 2)
  bert_recall = round(sum(bert_score["recall"]) / len(bert_score["recall"]), 2)
  bert_f1 = round(sum(bert_score["f1"]) / len(bert_score["f1"]), 2)

  return {"meteor": rounded_meteor,
          "bleu": rounded_bleu,
          "bertscore_precision": bert_precision,
          "bertscore_f1": bert_f1,
          "bertscore_recall": bert_recall,
          "simple_accuracy": accuracy,
          "bart_score": rounded_bart,
          "rogue_score": rouge_score}

model_name = "./models/" + model_type + "/" + model_type + "_" + dataset_name

early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001)


# define objective function for optuna optimization
def objective(trial, train_dataset, val_dataset):    
    # define hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    args = Seq2SeqTrainingArguments(
      model_name,
      evaluation_strategy="epoch",
      save_strategy= "epoch",
      learning_rate=learning_rate,
      per_device_train_batch_size=2,
      per_device_eval_batch_size=2,
      gradient_accumulation_steps=1,
      load_best_model_at_end = True,
      weight_decay=0.01,
      num_train_epochs=config["num_train_epochs"],
      predict_with_generate=True,
      eval_accumulation_steps=1,
      fp16=False,
      )

    trainer = Seq2SeqTrainer(
        model, 
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [early_stopping]
    )

    trainer.train()

    return trainer.state.log_history[-1]["eval_simple_accuracy"]

# define custom function to pass extra args to the objective function
func = lambda trial: objective(trial, dataset["train"], dataset["validation"])

# Create a study object and optimize the hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(func, n_trials=10)

# Print the best trial and its parameters
print(f'Best trial: {study.best_trial.number}')
print(f'Best accuracy: {study.best_trial.value:.4f}')
print('Best parameters:')
for key, value in study.best_trial.params.items():
    print(f'    {key}: {value}') 


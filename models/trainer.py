# -*- coding: utf-8 -*-

import evaluate
import json
import pandas as pd
import re
import torch
import sys

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

train_df = pd.read_csv(train_split_path)
val_df = pd.read_csv(val_split_path)
test_df = pd.read_csv(test_split_path)

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
bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-base")

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

args = Seq2SeqTrainingArguments(
    model_name,
    evaluation_strategy="epoch",
    save_strategy= "epoch",
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=config["num_train_epochs"],
    predict_with_generate=True,
    eval_accumulation_steps=1,
    fp16=False,
    load_best_model_at_end = True
    )

trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [early_stopping]
)

trainer.train()

eval_results = trainer.evaluate(dataset["test"])

trainer.save_model()

device = torch.device("cuda")
model.to(device)

conversation = """
AITA for embarrassing my husband in front of his friends?

I (26 F) am married to K (25 M) , i met him a few year back when my best friend and his best friend started dating, after that our friend groups kind of joined together. So all of my husband's friends are also my friends and all my friends are his friends.

I've know my best friend (A) for 15 years, we've always been extremely close and lived together at one point, she's literally part of my family, she comes to all of my family events, my siblings refer to her as their sister and she just been extremely involved in my life for years. Now, me and my husband have a 6 month old daughter. My dad is extremely wealthy so he helps us out financially until my daughter is old enough to go to school, then i will take a job at his company that pays well. My husband has a job that he works from 7am - 4 pm and then afterwards him and his friends often hangout at my house or at one of their house's.

Here's the problem, my husband's best friend cheated on my best friend a few weeks ago, she's completely heartbroken and since they lived together my husband and i agreed that it's okay that she stays with us until they work things out. She's not really up to seeing him just yet. A is a huge help with the baby, she's so good with her and the baby loves her. When my husband and his friends get home they normally talk until 8 pm ish so he doesn't help with the baby too much when he gets home, which i don't mind it's just nice to have a little help. Yesterday night my husband opened the front door , peaked his head in, looked right at me and A , rolled his eyes and walked out. Him and his friends came in a few moments later and were all being cold and rude to A all evening. After an hour or two my husband walks right up to me and A and says “does she always have to be here? I want to bring (friends name) here tonight.” And rolled his eyes, i responded “if i want to bring my best friend into the house my dad paid for, i can, and if i choose to have somebody help me with the baby that you're not looking after, it's going to be her.” His friends just sat there silently and he just walked away and sat back down . He's giving me the silent treatment and A left and is staying with her mom as she feels uncomfortable being in the house now. So AITA?
"""

inputs = tokenizer([conversation], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"].to(device), num_beams=2, min_length=25, max_length=config["token_sequence_length"])
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])



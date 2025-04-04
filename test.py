import torch
import argparse
import json
import os
import bitsandbytes as bnb
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, set_seed, read_data

parser = argparse.ArgumentParser(description="Inference the model on Instruction-tuning task")
parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--adapter_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--max_target_length", type=int, default=256)
parser.add_argument("--output_file", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
args = parser.parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Seed for reproducibility
set_seed(args.seed)

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    load_in_4bit= True,
    device_map='auto'
)

model = PeftModel.from_pretrained(model, args.adapter_name_or_path)

# load data
testFile = read_data(args.test_file)

# dataloader
test_dataloader = DataLoader(testFile, batch_size=1, collate_fn=lambda x: x)

gen_kwargs = {
    "max_new_tokens": args.max_target_length,
    "num_beams": 1,
}

results = []
model.eval()

for batch in tqdm(test_dataloader):
    instruction_text = batch[0]["instruction"]
    instruction_id = batch[0]["id"]

    prompt_text = get_prompt(instruction_text)

    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, **gen_kwargs)
        decoded_pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    if decoded_pred.startswith(prompt_text):
        decoded_pred = decoded_pred[len(prompt_text):].strip()

    result = {
        "id": instruction_id,
        "output": decoded_pred
    }
    results.append(result)

with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

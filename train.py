import torch
import argparse
import os
import bitsandbytes as bnb
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from utils import get_prompt, get_bnb_config, log, set_seed, read_data

parser = argparse.ArgumentParser(description="Instruction tuning on LoRA")
parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--max_source_length", type=int, default=1024)
parser.add_argument("--max_target_length", type=int, default=256)
parser.add_argument("--ignore_index", type=int, default=-100)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=int, default=0.1)
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate (after the potential warmup period) to use.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--logging_step", type=int, default=250)
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
args = parser.parse_args()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Seed for reproducibility
set_seed(args.seed)

# load tokenizer & model
bnb_config = get_bnb_config()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)

model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["self_attn.q_proj", "self_attn.v_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# load data
trainFile = read_data(args.train_file)

# dataset & dataloader
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction_text = get_prompt(item['instruction'])
        output_text = item['output']

        source = f"{self.tokenizer.bos_token}{instruction_text}"
        target = f"{output_text}{self.tokenizer.eos_token}"

        tokenized_source = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            add_special_tokens=False
        )
        
        tokenized_target = self.tokenizer(
            target,
            max_length=self.max_target_length,
            truncation=True,
            add_special_tokens=False
        )

        input_ids = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
        labels = torch.tensor([args.ignore_index] * len(tokenized_source['input_ids']) + tokenized_target['input_ids'])
        attention_mask = input_ids != self.tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def data_collator(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=args.ignore_index)

    data_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    return data_dict

train_dataset = TranslationDataset(trainFile, tokenizer)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)

# optimizer
optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# training
log('Strat training')
root = args.output_dir
for epoch in range(args.num_train_epochs):
    step = 1
    train_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        train_loss += loss.item()

        if step % args.logging_step == 0:
            log(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / args.logging_step:.3f}")
            train_loss = 0
            model.save_pretrained(os.path.join(root, f'checkpoint_{epoch}_{step}'))
import torch
import json
import os
import pandas as pd
import tiktoken
import tensorflow as tf
import sys
sys.path.append('/media/buct-5900x/Seagate-3/LZZ/LLM/build-gpt-scratch')
from torch.utils.data import DataLoader
from chapter6.SpamDataset import SpamDataset
from models.GPTModel import GPTModel
from utils import *
from gpt_download import load_gpt2_params_from_tf_ckpt
from chapter6.LoRA import LinearWithLoRA
from chapter6.train import train_classifier
from settings import BASE_CONFIG, model_configs

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_frac, valid_frac):
    df = df.sample(
        frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    valid_end = train_end + int(len(df) * valid_frac)
    train_df = df[:train_end]
    valid_df = df[train_end:valid_end]
    test_df = df[valid_end:]
    return train_df, valid_df, test_df

def replace_linear_with_lora(model, rank=4, alpha=1.0):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)

if __name__ == "__main__":
    df = pd.read_csv(
        "datasets/sms_spam_collection/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"]
    )
    # balanced_df = create_balanced_dataset(df)
    balanced_df = df
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, valid_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv('datasets/sms_spam_collection/train.csv', index=None)
    valid_df.to_csv('datasets/sms_spam_collection/valid.csv', index=None)
    test_df.to_csv('datasets/sms_spam_collection/test.csv', index=None)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
    csv_file='datasets/sms_spam_collection/train.csv',
    max_length=None,
    tokenizer=tokenizer,
    )
    valid_dataset = SpamDataset(
        csv_file='datasets/sms_spam_collection/valid.csv',
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    test_dataset = SpamDataset(
        csv_file='datasets/sms_spam_collection/test.csv',
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "../models/GPT2-124M/124M"
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)
    for param in model.parameters():
        param.requires_grad = False

    replace_linear_with_lora(model, rank=16, alpha=16.0)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    import time
    start_time = time.time()
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
        model, train_loader, valid_loader, optimizer, device, num_epochs, eval_freq=50, eval_iter=5,
        warmup_steps=0.2 * num_epochs * len(train_loader), init_lr=1e-5, min_lr=1e-5
    )
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f}")


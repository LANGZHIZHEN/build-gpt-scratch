import torch
from models.GPTModel import GPTModel
import tiktoken
from chapter2.dataloader import create_dataloader
from utils import *
import tensorflow as tf
import json
import os
from gpt_download import load_gpt2_params_from_tf_ckpt
from chapter5.train import train_model, train_model_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "num_layers": 12,
    "num_heads": 12,
    "emb_dim": 768,
    "context_length": 256,
    "dropout": 0.1,
    "num_classes": 2,
    'bias':True
}
def train():
    file_path = "datasets/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    train_ratio = 0.9
    split_idx = int(len(text) * train_ratio)
    train_data = text[:split_idx]
    val_data = text[split_idx:]
    train_loader = create_dataloader(train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'], 
                                    stride=GPT_CONFIG_124M['context_length'], shuffle=True, drop_last=True, num_workers=0)
    val_loader = create_dataloader(val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'], 
                                stride=GPT_CONFIG_124M['context_length'], shuffle=False, drop_last=True, num_workers=0)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    epoch = 50
    # train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, epoch,
    #                                                         eval_freq=5, eval_iter=1, start_context="Every effort moves you", tokenizer=tokenizer,
    #                                                         warmup_steps=0.2 * epoch * len(train_loader), init_lr=1e-5, min_lr=1e-5)
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, epoch,
                                                            eval_freq=5, eval_iter=1, start_context="Every effort moves you", tokenizer=tokenizer)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # },
    # "model_checkpoint.pth")

def predict():
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "../models/GPT2-124M/124M"
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    model =  GPTModel(GPT_CONFIG_124M)
    model.eval()
    load_weights_into_gpt(model, params)
    model.to(device)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("More pain, More gain", tokenizer).to(device),
        max_new_tokens=50,
        context_size=GPT_CONFIG_124M['context_length'],
        top_k=25,
        temperature=1.3
        )
    print(token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    train()
    
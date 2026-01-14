import torch
from models.GPTModel import GPTModel
import tiktoken
from datasets.dataloader import create_dataloader
from utils import *

def generate(model, idx, max_new_tokens, context_size,
             temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, k=top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, 
                                 torch.tensor(-float('Inf'), device=logits.device), 
                                 logits)
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.softmax(logits, dim=-1)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        valid_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, valid_loss

def generate_and_print_sample(model, tokenizer, start_context, device):
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, encoded, 50, context_size, 1.4, 15)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' ')) 
    model.train()
    

def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, step = 0, -1
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inputs.numel()
            step += 1

            if step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Epoch: {epoch}, Step: {step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        generate_and_print_sample(model, tokenizer, start_context, device)
    return train_losses, val_losses, track_tokens_seen

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "num_layers": 12,
    "num_heads": 12,
    "emb_dim": 768,
    "context_length": 256,
    "dropout": 0.1,
    "num_classes": 2,
    'bias':False
}

if __name__ == "__main__":
    file_path = "the-verdict.txt"
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    epoch = 50
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, epoch,
                                                            eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },
    "model_checkpoint.pth")
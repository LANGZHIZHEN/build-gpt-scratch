import math
import torch
from utils import text_to_token_ids, token_ids_to_text, generate, evaluate_model
from chapter5.calc_loss import calc_loss, calc_loss_loader


def generate_and_print_sample(model, tokenizer, start_context, device):
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, encoded, 50, context_size)
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
                    train_loss, val_loss = evaluate_model(calc_loss_loader, model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Epoch: {epoch}, Step: {step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        generate_and_print_sample(model, tokenizer, start_context, device)
    return train_losses, val_losses, track_tokens_seen

def train_model(model, train_loader, val_loader, optimizer, device, 
                       epochs, eval_freq, eval_iter, start_context, tokenizer,
                       warmup_steps, init_lr=3e-5, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen = [], [], []
    peak_lr = optimizer.param_groups[0]['lr']
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.2 * total_steps)
    lr_increment = peak_lr / warmup_steps
    tokens_seen, step = 0, -1
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            step += 1
            ##学习率预热与余弦衰退
            if step < warmup_steps:
                lr = init_lr + step * lr_increment
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(progress * math.pi))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            loss = calc_loss(inputs, targets, model, device)
            loss.backward()

            if step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tokens_seen += inputs.numel()
            if step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss, val_loss = evaluate_model(calc_loss_loader, model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Epoch: {epoch}, Step: {step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        generate_and_print_sample(model, tokenizer, start_context, device)
    return train_losses, val_losses, track_tokens_seen





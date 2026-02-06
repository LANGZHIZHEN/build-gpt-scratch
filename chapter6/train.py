from chapter6.calc_loss import calc_loss_batch, calc_loss_loader
from chapter6.calc_acc import calc_acc_loader
from utils import evaluate_model
import math
import torch

def train_classifier(model, train_loader, val_loader, optimizer, device, 
                       epochs, eval_freq, eval_iter,
                       warmup_steps, init_lr=1e-5, min_lr=1e-5):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    peak_lr = optimizer.param_groups[0]['lr']
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.2 * total_steps)
    lr_increment = peak_lr / warmup_steps
    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step < warmup_steps:
                lr = init_lr + global_step * lr_increment
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(progress * math.pi))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(calc_loss_loader, model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(valid_loss)
                print(f"Epoch {epoch}, step {global_step}, train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")
        train_acc = calc_acc_loader(train_loader, model, device, num_batches=eval_iter)
        valid_acc = calc_acc_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Epoch {epoch}, train acc {train_acc*100:.4f}% | valid acc {valid_acc*100:.4f}%")
        train_accs.append(train_acc)
        val_accs.append(valid_acc)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


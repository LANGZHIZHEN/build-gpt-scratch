import torch
from chapter6.calc_loss import calc_loss_batch, calc_loss_loader
from chapter6.calc_acc import calc_acc_loader
from utils import evaluate_model

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, 
                       epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

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


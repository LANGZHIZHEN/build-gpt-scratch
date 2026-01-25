import torch
def calc_acc_loader(data_loader, model, device, num_batches=None):
    model.eval()
    corr_preds, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        preds = logits.argmax(dim=-1)
        num_examples += preds.shape[0]
        corr_preds += (preds == target_batch).sum().item()
    return corr_preds / num_examples
        

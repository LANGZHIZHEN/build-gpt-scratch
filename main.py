import torch
from models.GPTModel import GPTModel
import tiktoken

def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

GPT_CONFIG = {
    "vocab_size": 50257,
    "num_layers": 12,
    "num_heads": 12,
    "emb_dim": 768,
    "context_length": 1024,
    "dropout": 0.1,
    "num_classes": 2,
    'bias':False
}

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
context = "Hello, I am"
encoded = tokenizer.encode(context)
encode_tensor = torch.tensor(encoded).unsqueeze(0)
model = GPTModel(GPT_CONFIG)
model.eval()
out = generate_text(
    model=model,
    idx=encode_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG['context_length']
)
decode_text = tokenizer.decode(out.squeeze(0).tolist())
print(decode_text)
import tiktoken
import torch
from model.gpt import *
import torch.nn.functional as F


checkpoint = torch.load('checkpoint.pth')
device = "cuda"

model = GPT(GPTConfig(vocab_size=50304)).to(device)

# Restore the model state
model.load_state_dict(checkpoint['model'])


enc = tiktoken.get_encoding('gpt2')
start = "I am going to tell you about the sport baseball. Here is what you need to know about baseball:"
tokens = enc.encode(start)
print(start)
x = torch.tensor(tokens, dtype=torch.long).to(device)
x = x.unsqueeze(0)
max_length = 10
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indicies, -1, ix)
        x = torch.cat((x, xcol), dim=1)
        #print(enc.decode(x))


tokens = x[-1].tolist()
print(enc.decode(tokens))
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from model.gpt import *
import sys
# code starts here

sys.path.append('model/')

device = "cuda"
model = GPT(GPTConfig(vocab_size=50304)).to(device)

torch.set_float32_matmul_precision('high')

max_lr = 3e-5
epochs = 5

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)

pretrained = torch.load('model/baseball_finetuned1.pth')

model.load_state_dict(pretrained['model'])

model.config = pretrained['config']

print('Loading data from baseball_qa.pkl')

with open('data/baseball data/baseball_qa.pkl', 'rb') as f:
    data = pickle.load(f)

for epoch in range(50):
    print(f"Epoch {epoch}")

    model.train()

    optimizer.zero_grad()
    loss_accum = 0.0

    for microstep in range(len(data)):
        x, y = torch.tensor(data[microstep][:-1]), torch.tensor(data[microstep][1:])
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= len(data)
        loss_accum += loss.detach()
        loss.backward()

    print(f"{epoch} Training loss: {loss_accum}")
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

checkpoint = {
    'model' : model.state_dict(),
    'config' : model.config,
}

torch.save(checkpoint, 'model/baseball_finetuned2.pth')
print("Saved Checkpoint")
        
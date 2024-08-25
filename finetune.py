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

max_lr = 3e-6
epochs = 5
val_check_freq = 50

B = 4
T = 1024

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)

pretrained = torch.load('model/finetuned.pth')

model.load_state_dict(pretrained['model'])

optimizer.load_state_dict(pretrained['optimizer'])

model.config = pretrained['config']

start_epoch = pretrained['epoch']
start_step = pretrained['step']

print('Loading data from qa_pairs.pkl')

with open('data/finetune data/qa_pairs.pkl', 'rb') as f:
    data = pickle.load(f)

train_data = data[:int(len(data) * 0.9)]
val_data = data[int(len(data) * 0.9):]

def validation_check(epoch, step):
    checkpoint = {
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch' : epoch,
        'step' : step,
        'config' : model.config,
    }
    torch.save(checkpoint, 'model/finetuned.pth')
    print("Saved Checkpoint")
    
    # validation code
    model.eval()
    with torch.no_grad():
        val_loss_accum = 0
        val_loss_steps = 2000
        for i in range(val_loss_steps):
            xy = val_data[i]
            x, y = torch.tensor(xy[:-1]), torch.tensor(xy[1:])
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= val_loss_steps
            val_loss_accum += loss.detach()
    print(f"{step}: Validation loss = {val_loss_accum.item()}")
    with open("validation_finetuning_loss_log.txt", "a") as f:
        f.write(f"epoch {epoch} step {step} val {val_loss_accum.item():.4f}\n")

batchsize = 1000
validation_freq = 10

for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch}")

    model.train()

    for iteration in range(start_step, len(train_data) // batchsize):
        if iteration % validation_freq == 0 and iteration != start_step:
            validation_check(epoch, iteration)

        optimizer.zero_grad()
        loss_accum = 0.0
        for microstep in range(batchsize):
            idx = iteration * batchsize + microstep
            x, y = torch.tensor(train_data[idx][:-1]), torch.tensor(train_data[idx][1:])
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss / batchsize
            loss_accum += loss.detach()
            loss.backward()
        print(f"{iteration} Training loss: {loss.item()}")
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from model.gpt import *
import os
import numpy as np
import sys
# code starts here

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

    def save_state(self, filepath):
        state = {
            'current_shard': self.current_shard,
            'current_position': self.current_position,
        }
        torch.save(state, filepath)

    def load_state(self, filepath):
        state = torch.load(filepath)
        self.current_shard = state['current_shard']
        self.current_position = state['current_position']
        self.tokens = self.load_tokens(self.shards[self.current_shard])

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

sys.path.append('model/')

device = "cuda"
model = GPT(GPTConfig(vocab_size=50304)).to(device)

torch.set_float32_matmul_precision('high')

max_lr = 1e-5
min_lr = max_lr * 0.1
warmup_steps = 300
max_steps = 19073
val_check_freq = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ration = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ration <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ration))
    return min_lr + coeff * (max_lr - min_lr)

total_batch_size = 2 ** 19
B = 4
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"grad accum steps = {grad_accum_steps}")
train_loader = DataLoaderLite(B, T, 'train')
val_loader = DataLoaderLite(B, T, 'val')

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)

load_from_checkpoint = True

init_step = 0

if load_from_checkpoint:
    checkpoint = torch.load('model/checkpoint.pth')

    # Restore the model state
    model.load_state_dict(checkpoint['model'])

    # Restore the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Restore the step
    init_step = checkpoint['step']

    model.config = checkpoint['config']
    train_loader.load_state('model/dataloader_state.pth')

for step in range(init_step, max_steps):
    if step % val_check_freq == 0 or step == max_steps - 1:
        checkpoint = {
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'config' : model.config,
            'step' : step,
        }
        torch.save(checkpoint, 'model/checkpoint.pth')
        train_loader.save_state('model/dataloader_state.pth')
        print("Saved Checkpoint")
        
        # validation code
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_steps = 1000
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= val_loss_steps
                val_loss_accum += loss.detach()
        print(f"{step}: Validation loss = {val_loss_accum.item()}")
        with open("validation_loss_log.txt", "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    t1 = time.time()
    model.train()
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    dt = (t2 - t1) * 1000
    tok_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t2 - t1)
    print(f"step {step} | loss: {loss_accum} | norm: {norm:.4f} | lr: {lr:.4e} | time: {dt} | tokens / second: {tok_per_sec}")

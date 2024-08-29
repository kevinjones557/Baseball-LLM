# Compare the performance between the baseball specific model and general model

import tiktoken
import torch
from model.gpt import *
import torch.nn.functional as F
from tqdm import tqdm
import sys

sys.path.append("model/")

print("Loading Model")
baseball_model_saved = torch.load('model/baseball_finetuned2.pth')
general_model_saved = torch.load('model/finetuned.pth')

device = "cuda"

baseball_model = GPT(GPTConfig(vocab_size=50304, device='cuda')).to(device)
general_model = GPT(GPTConfig(vocab_size=50304, device='cuda')).to(device)

# Restore the model state
baseball_model.load_state_dict(baseball_model_saved['model'])
general_model.load_state_dict(general_model_saved['model'])

enc = tiktoken.get_encoding('gpt2')

print("Loading Data")
with open("data/baseball data/baseball_validation_pairs.txt", "r", encoding="utf8") as f:
    lines = f.readlines()

question = ""
qa_pairs = []
for l in lines:
    if l == '\n':
        continue
    if l[0] == 'Q':
        question = l[3:].replace('\n', '')
    elif l[0] == 'A':
        answer = l[3:].replace('\n', '') + '<|endoftext|>'
        qa_pairs.append((enc.encode(question), enc.encode(answer, allowed_special={'<|endoftext|>'})))

print("Starting Comparison")
gen_prob_sum = 0.0
baseball_prob_sum = 0.0
count = 0
for pair in tqdm(qa_pairs):
    context = torch.tensor(pair[0]).to(device)
    context = context.unsqueeze(0)
    for i in range(len(pair[1])):
        with torch.no_grad():
            # run through the model
            baseball_logits, _ = baseball_model(context)
            general_logits, _ = general_model(context)
            baseball_logits = baseball_logits[:, -1, :]
            general_logits = general_logits[:, -1, :]
            baseball_probs = F.softmax(baseball_logits, dim=-1).squeeze(0)
            general_probs = F.softmax(general_logits, dim=-1).squeeze(0)

            #print(baseball_probs.shape)

            # check predicted prob with true value
            baseball_prob_sum += baseball_probs[pair[1][i]]
            gen_prob_sum += general_probs[pair[1][i]]
            count += 1

            next_char = torch.tensor([pair[1][i]]).unsqueeze(0).to(device)
            context = torch.cat((context, next_char), dim=1)

print(f"Baseball avg prob: {baseball_prob_sum / count}")
print(f"General avg prob: {gen_prob_sum / count}")


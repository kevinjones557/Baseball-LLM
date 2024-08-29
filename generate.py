import tiktoken
import torch
from model.gpt import *
import torch.nn.functional as F
import sys
import warnings

warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

sys.path.append('model/')

# Load the model from the saved .pth file and specify the device
saved_model = torch.load('model/baseball_finetuned2.pth')

device = "cuda"

model = GPT(GPTConfig(vocab_size=50304, device='cuda')).to(device)

# Restore the model state
model.load_state_dict(saved_model['model'])

# get an encoding object
enc = tiktoken.get_encoding('gpt2')
context = torch.tensor([], dtype=torch.long).to(device)
max_context_length = 1024

while True:
    print("User: ", end = '')
    start = input()
    new_context = enc.encode(start)
    # fill context from user prompt
    new_context = torch.tensor(new_context, dtype=torch.long).to(device)
    # add new context to previous context
    context = torch.cat((context, new_context), dim=0)
    print("\nAssistant: ", end='')
    while context.size(0) < 10000: # just in case the endoftext token is never hit
        # no backward pass so no need to maintain graph
        with torch.no_grad():
            # ensure the context_length is not exceeded
            if len(context) > max_context_length:
                context = context[-max_context_length:]
            # create a batch dimension
            context = context.unsqueeze(0)
            # run through the model
            logits, loss = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # only keep top 50 most likely tokens
            topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
            # sample from the probabilites
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indicies, -1, ix)
            # add the next token to the context
            context = torch.cat((context, xcol), dim=1)
            next_char = xcol.squeeze(0).tolist()
            # remove batch dimension so that the new context can be concatonated
            context = context.squeeze(0)
            # break if token is <|endoftext|>
            if xcol.squeeze(0).tolist() == [50256]:
                print("\n")
                break
            if next_char == [247]:
                continue
            if next_char == [447]:
                print("'", end='', flush=True)
                continue
            print(enc.decode(next_char), end='', flush=True)
            #time.sleep(0.05)

# Baseball Geared Large Language Model

## Overview
The goal of this project was to create and train a large language model from scratch using the Transformer architecture. I wanted to then tailor the model to answer questions related to baseball. I found it to be quite a challenge to collect a substantial amount of baseball specific data, but I was able to find some to train the model on. After comparing the baseball-tuned model with the regular model, it was clear that baseball themed questions were answered much better by the baseball-tuned model, but essentially all other types of questions were far better answered by the general model. 

## Model Design
To design the model, I followed the Transformer architecture outlined by the 'Attention is all you need' paper from 2017. I implemented this model in pytorch using only linear and embedding layers (i.e I did not use the Transformer block from pytorch) in order to develop a fundamental understanding of the architecture. I built the multi-headed self-attention using just these basic layers and created the rest of the architecture in the same way. To choose my hyperparameters, I followed OpenAI's design of GPT2-small. This architecture consists of 124 million parameters. I also used OpenAI's gpt2 tokenizer. Initially, I trained a custom tokenizer using the byte pair encoding algorithm in order to decrease the vocabulary size and consequently, the size of the model itself, but I found that without the many optimizations tiktoken implements, it was far to slow to tokenize large datsets.

## Pretraining the Model
I initialized the model with random weights and trained it from scratch. Using an NVIDIA 4070 GPU, I pretrained the model for several days on the fineweb-edu dataset from Hugging Face. I used the learning rate decay function as described in the GPT2 paper and trained it with a batch size of 2^19. Because of the limited memory space on my GPU, I was only able to process 2^12 tokens at a time. In order to fix this, I implemented gradient accumulation to simulate a larger batch size for training the model. Below is a graph of the validation loss throughout the pretraining of the model. The two visible stepdowns are when I manually lowered the learning rate.

![pretraining_loss](https://github.com/user-attachments/assets/1943d27e-6b54-4abf-b424-5708b302df47)

## Finetuning the Model
After I pretrained the model, I used the Nectar dataset from berkley-nest to finetune the model. I first preprocessed the data by adding the end-of-text token. I then trained the model for another several days with a batch size of 1000, again using gradiant accumulation. Throughout this training, the model begain to learn how to respond as an assistant by answering user's questions. Below is a graph of the validation loss throughout the finetuning of the model.

![fintuning_loss](https://github.com/user-attachments/assets/a16ad6e1-c5ac-4cd0-82be-36974755580d)

## Baseball Model

## Results

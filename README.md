# Baseball Geared Large Language Model

## Demo
https://github.com/user-attachments/assets/7fdcf113-cbc0-47ea-a493-4c27d0059519

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
After I had successfully finetuned a general model, I wanted to gear the model toward answering baseball questions. As I mentioned earlier, I found it extremely difficult to find pure text documents with relations to baseball. I ended up using the official baseball rules from 2021 to continue to pretrain the model. I trained it for 5 epochs on this data in an attempt to have the model learn about baseball but not forget about the previous training. I then used a small dataset of around 1000 baseball specific question answer pairs to finetune the model and increase its ability to answer baseball specific questions.

## Results
### Non-numerical Results
After training both the general model and the baseball specific model I compared their answers to different questions. The baseball model did successfully answer questions related to baseball far better than the general model did. However, the baseball model produced significantly worse results for essentially all other questions than the baseball model. Below are some examples comparing the two models on baseball and non-baseball questions. "User" designates the input I gave the model and "Assistant" designates the model's response.

I asked both models to explain what a fastball is in baseball. Here is the response from the baseball geared model:

![Fastball Good](https://github.com/user-attachments/assets/57e889fb-4e01-408b-82de-395289ca269f)

And from the general model:

![Screenshot 2024-08-25 161345](https://github.com/user-attachments/assets/e1804649-ce05-4df5-9528-4cd732c9e307)

Clearly, the model that is geared toward baseball provides a much better result. Though it is far from perfect, which is to be expected given the small size of the model, it does provide some facts about a fastball. However, this does take away from its ability to answer other questions. Here I asked both models to explain machine learning and its relation to LLMs. Here is the response from the baseball model. As you can see, it is trying to relate back to StatCast, the program that the MLB uses to keep track of statistics.

![Screenshot 2024-08-25 162330](https://github.com/user-attachments/assets/3b1b1047-28d8-4cad-b362-759310c45c4c)

And here is the response from the general model:

![Screenshot 2024-08-25 161857](https://github.com/user-attachments/assets/fb05d0f2-5278-4cb0-a5f5-c20196e0e4a8)

### Numerical Results
To numerically compare the baseball model against the generic model, I split the baseball question and answer pairs into training and validation sets. The validation set was used to compare both models performance. Neither model had seen these pairs before. To compare the models, I filled the context with the question and recorded each model's output probability corresponding to the correct next token. I then took the average of the probabilites for each model. The baseball model had an average probability of .43 for the correct token. The general model only had an average probability of .29 for the correct token. This means that the baseball model observed a 50 % larger probability for the correct token.

## Outcomes
Though these responses are far from perfect given the small nature of the model, they do represent the difference between a model that has been geared toward a topic and one that has not. The baseball specific model definetly answers baseball questions far better than the general model. However, it does lose its genearlity because of its specific finetuning. Overall though, both consistently generate grammatically correct content that generally makes sense.

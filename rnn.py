import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) 
        rnn_outputs, hidden = self.rnn(inputs) #seq_length*bs*hid_dim
        # print (rnn_outputs.shape)
        # [to fill] obtain output layer representations
        final_output = self.W(rnn_outputs) #seq_length*bs*5
        # print (final_output.shape)
        # [to fill] sum over output 
        sum_output = final_output.sum(dim=0) #bs*5
        # print (sum_output.shape)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(sum_output)

        return predicted_vector


# def load_data(train_data, val_data):
#     with open(train_data) as training_f:
#         training = json.load(training_f)
#     with open(val_data) as valid_f:
#         validation = json.load(valid_f)

#     tra = []
#     val = []
#     for elt in training:
#         tra.append((elt["text"].split(),int(elt["stars"]-1)))
#     for elt in validation:
#         val.append((elt["text"].split(),int(elt["stars"]-1)))
#     return tra, val

#rectified version
def load_data(train_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    tra = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "test.json", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='runs/exp1', help="path to save checkpoint directory")
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print("========== Loading data ==========")
    # train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    train_data = load_data(args.train_data)
    valid_data = load_data(args.val_data)
    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    train_losses = []
    validation_losses = []
    train_accuracy = []
    val_accuracy = []
    best_epoch = 0
    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
       
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        # print(loss_total/loss_count)
        total_loss = loss_total/loss_count
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training loss for epoch {}: {}".format(epoch + 1,total_loss))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        train_accuracy.append(correct/total*100)
        train_losses.append(total_loss.detach().numpy())
        trainning_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0
        # random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        # valid_data = valid_data
        loss = 0
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            loss = loss+example_loss
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation loss for epoch {}: {}".format(epoch + 1,loss/total))
        validation_accuracy = correct/total
        validation_losses.append(loss/total)
        val_accuracy.append(validation_accuracy*100)
        
        if validation_accuracy <= last_validation_accuracy and trainning_accuracy >= last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy
            best_epoch = epoch
            ckpt_path = f"{args.ckpt_dir}/best_epoch{epoch}_acc{validation_accuracy:.2f}.pt"
            torch.save(model.state_dict(),ckpt_path)

        epoch += 1


import csv
metrics_path = f"{args.ckpt_dir}/metrics.csv"
with open(metrics_path,"w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["epoch", "train_loss", "train_acc(%)", "val_loss", "val_acc(%)"])
    for i in range(len(train_losses)):
        w.writerow([
            i + 1,
            f"{(train_losses[i]):.6f}",
            f"{(train_accuracy[i]):.6f}",
            f"{(validation_losses[i]):.6f}",
            f"{(val_accuracy[i]):.6f}",
        ])

   
#Loaded best model
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)
print(f"Loaded best checkpoint from epoch {best_epoch} with val_acc={last_validation_accuracy:.2f}")

#Code to test the data
test_data = load_data(args.test_data)
correct = 0
total = 0
loss = 0
print("Testing started..")

for input_words, gold_label in tqdm(test_data):
    input_words = " ".join(input_words)
    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                in input_words]

    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
    output = model(vectors)
    example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

    predicted_label = torch.argmax(output)
    correct += int(predicted_label == gold_label)
    total += 1
    loss = loss+example_loss
    # print(predicted_label, gold_label)
print("Test completed")
print("Test accuracy : {}".format(correct / total))
print("Test loss : {}".format(loss/total))

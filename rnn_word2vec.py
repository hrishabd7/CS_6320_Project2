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
from gensim.models import KeyedVectors
import sys

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h, embedding_matrix=None,
                 layers=1,activation='tanh',train_embd=False):  # Add relevant parameters
        
        super(RNN, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=not train_embd
            )
            input_dim = embedding_matrix.shape[1]
        else:
            self.embedding=None

        self.h = h
        self.numOfLayer = layers
        self.rnn = nn.RNN(input_dim,h,self.numOfLayer, nonlinearity=activation)
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.sum = True

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        if self.embedding is not None:
            # print (inputs.shape)
            emb = self.embedding(inputs) 
            # print (emb.shape)
            rnn_outputs, hidden = self.rnn(emb)  
        else:
            rnn_outputs, hidden = self.rnn(inputs) #seq_length*bs*hid_dim

        # [to fill] obtain output layer representations      
        if self.sum:
            final_output = self.W(rnn_outputs) #seq_length*bs*5
            # [to fill] sum over output 
            sum_output = final_output.sum(dim=0) #bs*5
            # [to fill] obtain probability dist.
            predicted_vector = self.softmax(sum_output)
        else:
            final_output = self.W(hidden).sum(dim=0)
            # print (final_output.shape)
            predicted_vector = self.softmax(final_output)
            # print (predicted_vector.shape)
        return predicted_vector

def load_data(train_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    tra = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra

def normalize(tok):
    tok = tok.translate(str.maketrans("", "", string.punctuation))
    tok = tok.lower()
    return tok

def make_vocab(data):
    
    vocab = set()
    for document, _ in data:
        for word in document:  
            vocab.add(normalize(word))
    return vocab 

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

def create_word_emb(w2v, vocab_size,word2index,emb_dim):
    
    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    for word, idx in word2index.items():
        if word in w2v:
            embedding_matrix[idx] = w2v[word]
        else:
            # random init for OOV
            embedding_matrix[idx] = np.random.normal(scale=1, size=(emb_dim,))
    return embedding_matrix

def load_word2_vec(filename):
    w2v = KeyedVectors.load_word2vec_format(
        filename,
        binary=True
    )
    # print (w2v)
    return w2v

def load_glove(glove_file):
    
    glove = {}
    with open(glove_file, "r") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype="float32")
            glove[word] = vec
    return glove

import matplotlib.pyplot as plt

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, epochs, save_dir=None):
    print (epochs)
    n = len(train_loss)
    assert len(val_loss) == n and len(train_acc) == n and len(val_acc) == n, "All lists must be same length."

    # Figure 1: Loss 
    plt.figure()
    print (val_loss)
    print (train_loss)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Figure 2: Accuracy 
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "test.json", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='runs/rnn1', help="path to save checkpoint directory")
    parser.add_argument('--word_embedding', type=str, default='word2vec', help="path to save checkpoint directory")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--minibatch_size', type=int, default=16, help="minibatch_size")
    parser.add_argument('--layers', type=int, default=1, help="layers")
    parser.add_argument('--activation', type=str, default='tanh', help="activation")
    parser.add_argument('--train_embedding', action='store_true', help="path to save checkpoint directory")

    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("========== Loading data ==========")
    train_data = load_data(args.train_data)[:4000]
    valid_data = load_data(args.val_data)
    test_data = load_data(args.test_data)
    if args.word_embedding=='word2vec':
        w2v = load_word2_vec('word_embeddings/GoogleNews-vectors-negative300.bin')
        emb_dim = w2v.vector_size
    elif args.word_embedding=='glove100d':
        w2v = load_glove('word_embeddings/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt')
        emb_dim = w2v['and'].shape[0]
    else:
        print (f"Print the right embedding")
        sys.exit(0)

    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    vocab_size = len(vocab)
    word_embedding = create_word_emb(w2v, vocab_size,word2index,emb_dim)

    ## build embdeeing matrix
    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim, 
                layers=args.layers,
                embedding_matrix=word_embedding,\
                activation=args.activation,\
                train_embd =args.train_embedding).to(device)  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    stopping_condition = False
    epoch = 0
    best_validation_accuracy = 0
    best_val_loss = 100
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

        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = args.minibatch_size
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
                vectors = [word2index.get(i.lower(), word2index[unk]) for i in input_words ]
                
                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), -1).to(device)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]).to(device))

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

        total_loss = loss_total/loss_count
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training loss for epoch {}: {}".format(epoch + 1,total_loss))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        train_accuracy.append(correct/total*100)
        train_losses.append(total_loss.detach().cpu().numpy())
        trainning_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0

        print("Validation started for epoch {}".format(epoch + 1))
        loss = 0
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

            # Look up word embedding dictionary
            vectors = [word2index.get(i.lower(), word2index[unk]) for i in input_words ]
                
            vectors = torch.tensor(vectors).view(len(vectors), -1).to(device)
            output = model(vectors)
            example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]).to(device))

            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            loss = loss+example_loss

        val_loss = loss/total
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation loss for epoch {}: {}".format(epoch + 1,val_loss))
        validation_accuracy = correct/total
        validation_losses.append(val_loss.detach().cpu().numpy())
        val_accuracy.append(validation_accuracy*100)


        if val_loss<best_val_loss:
            count=0
            best_val_loss=val_loss
        else:
            count+=1
            if count==3:
                stopping_condition=True
                print("Training done to avoid overfitting!")
                print("Best validation accuracy is:", last_validation_accuracy)
        if validation_accuracy>best_validation_accuracy:
            best_epoch = epoch
            ckpt_path = f"{args.ckpt_dir}/best_epoch{best_epoch}_acc{validation_accuracy:.2f}.pt"
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(),ckpt_path)
            count=0
        epoch+=1
plot_training_curves(train_losses, validation_losses, train_accuracy, val_accuracy, np.arange(len(val_accuracy)), save_dir=args.ckpt_dir)
        
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

   
state_dict = torch.load(ckpt_path, map_location="cpu",weights_only=True)
model.load_state_dict(state_dict)
print(f"Loaded best checkpoint from epoch {best_epoch} with val_acc={best_validation_accuracy:.2f}")

#Code to test the data
test_data = load_data(args.test_data)
correct = 0
total = 0
loss = 0
print("Testing started..")

for input_words, gold_label in tqdm(test_data):
    input_words = " ".join(input_words)
    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
    vectors = [word2index.get(i.lower(), word2index[unk]) for i in input_words ]
                
    vectors = torch.tensor(vectors).view(len(vectors), -1).to(device)
    output = model(vectors)
    example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]).to(device))

    predicted_label = torch.argmax(output)
    correct += int(predicted_label == gold_label)
    # print (f"Predicted",predicted_label)
    # print (f"Gold label",gold_label)
    # print (".....")
    total += 1
    loss = loss+example_loss
    # print(predicted_label, gold_label)
print("Test completed")
print("Test accuracy : {}".format(correct / total))
print("Test loss : {}".format(loss/total))
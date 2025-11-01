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
from argparse import ArgumentParser


unk = '<UNK>'

minibatch_size = 32
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h, num_layers=1):
        super(FFNN, self).__init__()
        self.h = h
        self.num_layers = max(1, int(num_layers))
        self.output_dim = 5

        # Build hidden layers
        self.hidden_layers = nn.ModuleList()
        in_features = input_dim
        # first hidden layer
        self.hidden_layers.append(nn.Linear(in_features, h))
        # additional hidden layers (h -> h)
        for _ in range(1, self.num_layers):
            self.hidden_layers.append(nn.Linear(h, h))

        # Output layer
        self.W2 = nn.Linear(h, self.output_dim)

        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.softmax = nn.LogSoftmax(dim=-1) # Specify dim to avoid deprecation warning; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Pass through hidden layers with activation
        x = input_vector
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Obtain output layer representation
        output_representation = self.W2(x)

        # Obtain probability dist.
        predicted_vector = self.softmax(output_representation)
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
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

# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

#rectified version
def load_data(train_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    tra = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra

#rectify this
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

import matplotlib.pyplot as plt

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, epochs, save_dir=None):
    print (epochs)
    n = len(train_loss)
    assert len(val_loss) == n and len(train_acc) == n and len(val_acc) == n, "All lists must be same length."

    # Figure 1: Loss 
    plt.figure()
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
    parser.add_argument("-nl", "--num_layers", type=int, default=1, help = "number of hidden layers")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "test.json", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='runs/exp1', help="path to save checkpoint directory")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam", "adamw", "rmsprop", "adagrad"])
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "leakyrelu", "tanh", "sigmoid", "elu", "gelu"])
    
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data = load_data(args.train_data)
    valid_data = load_data(args.val_data)
    # train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    print (f"Vocab length is {len(vocab)}")
    print (f"Length of train data is {len(train_data)}")
    print (f"Length of validation data is {len(valid_data)}")

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim, num_layers=args.num_layers)

    # choose activation
    if args.activation == "relu":
        model.activation = nn.ReLU()
    elif args.activation == "leakyrelu":
        model.activation = nn.LeakyReLU()
    elif args.activation == "tanh":
        model.activation = nn.Tanh()
    elif args.activation == "sigmoid":
        model.activation = nn.Sigmoid()
    elif args.activation == "elu":
        model.activation = nn.ELU()
    elif args.activation == "gelu":
        model.activation = nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {args.activation}")

    # choose optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print("========== Training for {} epochs ==========".format(args.epochs))
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []
    best_validation_accuracy = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        #defined total loss over the epoch
        total_loss = 0
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            total_loss+= loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        total_loss = total_loss/(minibatch_size*(N//minibatch_size))
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training loss for epoch {}: {}".format(epoch + 1,total_loss))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        train_accuracy.append(correct/total*100)
        train_losses.append(total_loss.detach().numpy())

        loss = None
        correct = 0
        total = 0
        #defined loss for the validation
        total_loss = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            total_loss+= loss
            loss = loss / minibatch_size
        
        total_loss = total_loss/(minibatch_size*(N//minibatch_size))
        val_acc = (correct/total) * 100.0

        validation_accuracy.append(val_acc)
        validation_losses.append(total_loss.detach().numpy())
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation loss for epoch {}: {}".format(epoch + 1,total_loss))        
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        
        # added code for getting best model on validation set
        if val_acc>best_validation_accuracy:
            best_validation_accuracy = val_acc
            best_epoch = epoch + 1
            ckpt_path = f"{args.ckpt_dir}/best_epoch{best_epoch}_acc{val_acc:.2f}.pt"
            torch.save(model.state_dict(),ckpt_path)
            # torch.save(torch.load(ckpt_path), ckpt_dir / "best_model.pt")

plot_training_curves(train_losses, validation_losses, train_accuracy, validation_accuracy, np.arange(args.epochs), save_dir=args.ckpt_dir)
## addded all the metrics
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
            f"{(validation_accuracy[i]):.6f}",
        ])

#Loaded best model
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)
print(f"Loaded best checkpoint from epoch {best_epoch} with val_acc={best_validation_accuracy:.2f}")

#Code to test the data
test_data = load_data(args.test_data)
test_data = convert_to_vector_representation(test_data, word2index)
print (f"Length of test data is: {len(test_data)}")
model.eval()

loss = None
correct = 0
total = 0
total_loss = 0
start_time = time.time()
print("Testing started") 
N = len(test_data) 
for minibatch_index in tqdm(range(N // minibatch_size)):
    optimizer.zero_grad()
    loss = None
    for example_index in range(minibatch_size):
        input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        correct += int(predicted_label == gold_label)
        total += 1
        example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
        if loss is None:
            loss = example_loss
        else:
            loss += example_loss
    total_loss+=loss
    loss = loss / minibatch_size
total_loss = total_loss/(minibatch_size*(N//minibatch_size))
print("Test completed ")
print("Test loss: {}".format(total_loss))
print("Test accuracy: {}".format(correct / total))
print("Test time: {}".format(time.time() - start_time))
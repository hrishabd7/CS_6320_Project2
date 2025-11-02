import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import random
import os
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h,layers=1,activation='tanh',bidirectional=False,last=False):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = layers
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity=activation,\
                          bidirectional=bidirectional)
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.last = last
        self.directions = 2 if bidirectional else 1
        self.W = nn.Linear(self.directions*h, 5)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) 
        rnn_outputs, hidden = self.rnn(inputs) #seq_length*bs*hid_dim
        # [to fill] obtain output layer representations
        if not self.last:
            final_output = self.W(rnn_outputs) #seq_length*bs*5
            sum_output = final_output.sum(dim=0) #bs*5
            predicted_vector = self.softmax(sum_output)
        else:
            final_output = self.W(hidden).sum(dim=0)
            predicted_vector = self.softmax(final_output)
            
        return predicted_vector


#rectified version
def load_data(train_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    tra = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra
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
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--minibatch_size', type=int, default=16, help="minibatch_size")
    parser.add_argument('--layers', type=int, default=1, help="layers")
    parser.add_argument('--activation', type=str, default='tanh', help="activation")
    parser.add_argument('--dropout', type=int, default=1, help="dropout")
    parser.add_argument('--bidirectional', action="store_true", help="whether to use bidirectional")
    parser.add_argument('--last', action="store_true", help="whether to use bidirectional")

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print("========== Loading data ==========")
    # train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    train_data = load_data(args.train_data)
    valid_data = load_data(args.val_data)
    test_data = load_data(args.test_data)
    
    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim, args.layers,activation=args.activation,\
                bidirectional=args.bidirectional,last=args.last).to(device)  
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    best_validation_accuracy = 0
    best_val_loss = 10
    last_train_accuracy = 0
    last_validation_accuracy = 0
    train_losses = []
    validation_losses = []
    train_accuracy = []
    val_accuracy = []
    best_epoch = 0
    count =0
    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
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
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
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
        # print(loss_total/loss_count)
        total_loss = loss_total/loss_count
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training loss for epoch {}: {}".format(epoch + 1,total_loss))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        train_accuracy.append(correct/total*100)
        train_losses.append(total_loss.detach().cpu().numpy())

        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))
        loss = 0
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
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
            ckpt_path = f"{args.ckpt_dir}/best_epoch{epoch}_acc{validation_accuracy:.2f}.pt"
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(),ckpt_path)
            count=0

        # if validation_accuracy <= last_validation_accuracy and trainning_accuracy >= last_train_accuracy:
        #    if count==3:
        #         stopping_condition=True
        #     print("Training done to avoid overfitting!")
        #     print("Best validation accuracy is:", last_validation_accuracy)
            # else:
            # last_validation_accuracy = validation_accuracy
            # last_train_accuracy = trainning_accuracy
        
        # else:
        #     count+=1
        #     if count==3:
        #         stopping_condition=True
        #         print("Training done to avoid overfitting!")
        #         print("Best validation accuracy is:", last_validation_accuracy)
            
        epoch += 1

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

   
#Loaded best model
# ckpt_path = f"{args.ckpt_dir}/best_epoch1_acc0.55.pt"
state_dict = torch.load(ckpt_path, map_location="cpu",weights_only=True)
model.load_state_dict(state_dict)
model.eval()
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
    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                in input_words]

    vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
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

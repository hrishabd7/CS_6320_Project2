# CS_6320_Project2

End-to-end training scripts for a feed-forward neural network (FFNN) and an RNN using PyTorch, plus utilities to plot results across different learning rates and batch sizes.

## Prerequisites
- Python 3.8+
- macOS (tested), bash shell

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data
Place your data JSON files in the project root (or pass absolute paths):
- `training.json` / `training_new.json`
- `validation.json` / `validation_new.json`
- `test.json`

RNN also expects `word_embedding.pkl` in the project root. If running for word2vec and glove embedding, download the embedding files.

- Word2Vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
- Glove100d: https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip

## Train and Evaluate

### FFNN
Required args: `--hidden_dim`, `--epochs`, `--train_data`, `--val_data`
Optional: `--test_data`, `--ckpt_dir`, `--optimizer`, `--activation`, `--do_train`, `--numlayers`

Example:
```bash
python ffnn.py -hd 128 -e 10 \
  --num_layers 1 \
  --train_data training_new.json \
  --val_data validation_new.json \
  --test_data test.json \
  --ckpt_dir runs/ffnn_exp1 \
  --optimizer adam \
  --activation relu \
  --do_train
```

Supported values:
- Optimizer: `sgd`, `adam`, `adamw`, `rmsprop`, `adagrad`
- Activation: `relu`, `leakyrelu`, `tanh`, `sigmoid`, `elu`, `gelu`

Artifacts:
- Best checkpoint is saved in `--ckpt_dir` as `best_epoch<epoch>_acc<val_acc>.pt`
- Metrics are written to `<ckpt_dir>/metrics.csv` (tab-delimited):
  `epoch\ttrain_loss\ttrain_acc(%)\tval_loss\tval_acc(%)`

### RNN
Required args: `--hidden_dim`, `--epochs`, `--train_data`, `--val_data`
Optional: `--test_data`, `--ckpt_dir`, `--do_train`, `--num_layers`, `--lr`, `--minibatch_size`, `--activation {relu,tanh}`, `--bidirectional`, `--last`

Example:
```bash
python rnn.py -hd 128 -e 8 \
  --num_layers 1 \
  --train_data training_new.json \
  --val_data validation_new.json \
  --test_data test.json \
  --ckpt_dir runs/rnn_exp1 \
  --do_train
```

Example run that gave the best performance with word_embeddings.pkl
```bash
python rnn.py \
  --hidden_dim 256 \
  --epochs 40 \
  --train_data data/training_new.json \
  --val_data data/validation_new.json \
  --test_data data/test.json \
  --ckpt_dir runs/rnn1 \
  --lr 0.00001 \
  --minibatch_size 32
```

Optional parameters:
- `--activation {relu,tanh}`: activation function (default: `tanh`).
- `--bidirectional`: enable a bidirectional RNN.
- `--last`: use only the last hidden state for classification (no summing across all time steps).

Note: Ensure `word_embedding.pkl` is present in the project root.

### RNN with Word2Vec/Glove variant
To run RNN using a Word2Vec/Glove100 embedding, run 'rnn_word2vec.py` that has an option to train embeddings. Below is the example command:

To train the embedding as well (our best setting):
```bash
python rnn_word2vec.py \
  --hidden_dim 256 \
  --epochs 40 \
  --train_data data/training_new.json \
  --val_data data/validation_new.json \
  --test_data data/test.json \
  --ckpt_dir runs/word_embeddings/rnn2 \
  --lr 0.00001 \
  --minibatch_size 32 \
  --word_embedding word2vec \
  --train_embedding
```

To not train the embedding (use pretrained as-is):
```bash
python rnn_word2vec.py \
  --hidden_dim 256 \
  --epochs 40 \
  --train_data data/training_new.json \
  --val_data data/validation_new.json \
  --test_data data/test.json \
  --ckpt_dir runs/word_embeddings/rnn3 \
  --lr 0.00001 \
  --minibatch_size 32 \
  --word_embedding word2vec
```


## Imports Used

Third-party packages (installed via `requirements.txt`):
- `torch` (PyTorch) — `torch`, `torch.nn`, `torch.optim`
- `numpy`
- `matplotlib`
- `tqdm`
- `gensim`

Standard library:
- `os`, `json`, `argparse`, `time`, `random`, `math`, `string`, `csv`, `pickle`, `re`, `typing`, `collections.Counter`

## VS Code Tips
- Select the interpreter: Command Palette → “Python: Select Interpreter” → choose the `.venv` in this folder.
- If plots should save without opening windows (headless), run with:
  ```bash
  MPLBACKEND=Agg python ffnn.py ...
  ```
- To be explicit about the venv interpreter (ffnn):
  ```bash
  ./.venv/bin/pythonpython ffnn.py -hd 32 -e 50 --train_data training_new.json --val_data validation_new.json --test_data test.json --ckpt_dir runs/exp1 --optimizer adam --activation elu --do_train
  ```
- To be explicit about the venv interpreter (rnn):
  ```bash
  ./.venv/bin/pythonpython rnn.py -hd 256 -e 40 --train_data data/training_new.json --val_data data/validation_new.json --test_data data/test.json --ckpt_dir runs/rnn1 --lr 1e-5 --minibatch_size 32
  ```

## Troubleshooting
- "Imports not resolved" in VS Code: ensure the project venv is selected, or reload window after activation.
- If a plotting script skips a run, verify the folder naming pattern and that `metrics.csv` exists and is tab-delimited with the expected columns.

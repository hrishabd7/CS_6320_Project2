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

RNN also expects `word_embedding.pkl` in the project root.

## Train and Evaluate

### FFNN
Required args: `--hidden_dim`, `--epochs`, `--train_data`, `--val_data`
Optional: `--test_data`, `--ckpt_dir`, `--optimizer`, `--activation`, `--do_train`

Example:
```bash
python ffnn.py -hd 128 -e 10 \
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
Optional: `--test_data`, `--ckpt_dir`, `--do_train`

Example:
```bash
python rnn.py -hd 128 -e 8 \
  --train_data training.json \
  --val_data validation.json \
  --test_data test.json \
  --ckpt_dir runs/rnn_exp1 \
  --do_train
```

Note: Ensure `word_embedding.pkl` is present in the project root.


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
- To be explicit about the venv interpreter:
  ```bash
  ./.venv/bin/python ffnn.py -hd 128 -e 10 --train_data training_new.json --val_data validation_new.json --do_train
  ```

## Troubleshooting
- "Imports not resolved" in VS Code: ensure the project venv is selected, or reload window after activation.
- If a plotting script skips a run, verify the folder naming pattern and that `metrics.csv` exists and is tab-delimited with the expected columns.

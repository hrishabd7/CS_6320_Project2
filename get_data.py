import json
import random
from collections import Counter

random.seed(42)
field = "stars"
REQ_COUNT = 320

def count_ratings(data):
    c = Counter()
    for item in data:
        c[item[field]] += 1
    return c

def select_remove_random(data, label, needed):

    idxs = [i for i, item in enumerate(data) if item.get(field) == label]

    chosen_positions = set(random.sample(idxs, needed))

    picked = []
    remaining = []
    for i, item in enumerate(data):
        if i in chosen_positions:
            picked.append(item)
        else:
            remaining.append(item)

    return picked, remaining

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


train_data = read_file("./data/training_new2.json")
val_data = read_file("./data/validation_new2.json")

train_counts = count_ratings(train_data)
val_counts = count_ratings(val_data)
print (train_counts)
print (val_counts)
all_labels = set(train_counts) | set(val_counts)

for label in all_labels:
    curr_val = val_counts.get(label, 0)
    if curr_val < REQ_COUNT:
        need = REQ_COUNT - curr_val
        picked, train_data = select_remove_random(train_data, label, need)
        val_data.extend(picked)
        val_counts[label] = curr_val + len(picked)

write_file("./data/training_new2.json", train_data)
write_file("./data/validation_new2.json", val_data)

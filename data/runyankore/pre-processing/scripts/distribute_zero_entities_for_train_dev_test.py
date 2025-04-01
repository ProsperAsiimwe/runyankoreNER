import os
import random

# Paths for input and output
DATA_DIR = os.path.join(os.path.dirname(__file__), "../")
without_entities_file = os.path.join(DATA_DIR, "sentences_without_entities.txt")

# Existing dataset files
train_file = os.path.join(DATA_DIR, "train.txt")
dev_file = os.path.join(DATA_DIR, "dev.txt")
test_file = os.path.join(DATA_DIR, "test.txt")

# Function to read sentences from a CoNLL-style file
def read_sentences(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append("\n".join(current_sentence))
                    current_sentence = []
            else:
                current_sentence.append(line)
        if current_sentence:
            sentences.append("\n".join(current_sentence))
    return sentences

# Load existing dataset to prevent duplicates
def load_existing_sentences(paths):
    existing = set()
    counts = {}
    for path in paths:
        if os.path.exists(path):
            sents = read_sentences(path)
            existing.update(sents)
            counts[path] = len(sents)
    return existing, counts

# Load current sentences in datasets for summary
existing_sentences, pre_counts = load_existing_sentences([train_file, dev_file, test_file])

# Load and shuffle non-entity sentences
without_entities = read_sentences(without_entities_file)
random.shuffle(without_entities)

# Remove duplicates
without_entities = [s for s in without_entities if s not in existing_sentences]

# Split into train/dev/test
total = len(without_entities)
train_split = int(total * 0.5)
dev_split = int(total * 0.25)

train_without = without_entities[:train_split]
dev_without = without_entities[train_split:train_split + dev_split]
test_without = without_entities[train_split + dev_split:]

# Helper to append with proper spacing
def append_sentences(file_path, sentences):
    with open(file_path, "a", encoding="utf-8") as outfile:
        if os.path.getsize(file_path) > 0:
            outfile.write("\n\n")
        outfile.write("\n\n".join(sentences) + "\n\n")

# Append to datasets
append_sentences(train_file, train_without)
append_sentences(dev_file, dev_without)
append_sentences(test_file, test_without)

# Load new totals for summary
post_counts = {
    train_file: len(read_sentences(train_file)),
    dev_file: len(read_sentences(dev_file)),
    test_file: len(read_sentences(test_file))
}

# Summary
print("\n✅ Sentences without entities added:")
print(f"   ➕ Train: {len(train_without)}")
print(f"   ➕ Dev:   {len(dev_without)}")
print(f"   ➕ Test:  {len(test_without)}")

print("\n📊 Dataset summary:")
print(f"{'Split':<10}{'Before':>10}{'After':>10}{'Change':>10}")
for path, name in [(train_file, "Train"), (dev_file, "Dev"), (test_file, "Test")]:
    before = pre_counts.get(path, 0)
    after = post_counts.get(path, 0)
    change = after - before
    print(f"{name:<10}{before:>10}{after:>10}{change:>10}")

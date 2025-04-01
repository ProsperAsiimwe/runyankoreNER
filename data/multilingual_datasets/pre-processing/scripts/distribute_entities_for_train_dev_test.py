import os
import random
from collections import defaultdict

# Paths for input and output
DATA_DIR = os.path.join(os.path.dirname(__file__), "../")
input_file = os.path.join(DATA_DIR, "sentences_with_entities.txt")
train_file = os.path.join(DATA_DIR, "train.txt")
dev_file = os.path.join(DATA_DIR, "dev.txt")
test_file = os.path.join(DATA_DIR, "test.txt")

# Entity labels to focus on
entity_labels = {"B-PER", "B-ORG", "B-DATE", "B-LOC"}

# Function to read sentences from the CoNLL file
def read_sentences(file_path):
    sentences = []
    current_sentence = []
    
    with open(file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        if current_sentence:
            sentences.append(current_sentence)
    return sentences

# Function to extract labels in a sentence
def extract_labels(sentence):
    return {line.split()[-1] for line in sentence if len(line.split()) > 1}

# Read and shuffle all sentences
all_sentences = read_sentences(input_file)
random.shuffle(all_sentences)
print(f"🔍 Total sentences: {len(all_sentences)}")

# Create label-to-sentences mapping
label_to_sentences = defaultdict(list)
sentence_to_labels = {}

for sentence in all_sentences:
    labels = extract_labels(sentence)
    matched = entity_labels & labels
    if matched:
        sentence_str = "\n".join(sentence)
        for label in matched:
            label_to_sentences[label].append(sentence_str)
        sentence_to_labels[sentence_str] = matched

# Track used sentences
used_sentences = set()
train_sentences, dev_sentences, test_sentences = [], [], []

# Distribute based on each label without duplication
for label in sorted(entity_labels):  # sort for consistency
    candidates = [s for s in label_to_sentences[label] if s not in used_sentences]
    random.shuffle(candidates)

    total = len(candidates)
    n_train = int(total * 0.5)
    n_dev = int(total * 0.25)

    train, dev, test = candidates[:n_train], candidates[n_train:n_train + n_dev], candidates[n_train + n_dev:]

    train_sentences.extend(train)
    dev_sentences.extend(dev)
    test_sentences.extend(test)

    used_sentences.update(train + dev + test)

    print(f"✅ {label}: total={total}, train={len(train)}, dev={len(dev)}, test={len(test)}")

# Save to files
def save_sentences(path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sentences) + "\n\n")

save_sentences(train_file, train_sentences)
save_sentences(dev_file, dev_sentences)
save_sentences(test_file, test_sentences)

print(f"✅ Splitting complete!")
print(f"📁 Train: {len(train_sentences)} ➝ {train_file}")
print(f"📁 Dev: {len(dev_sentences)} ➝ {dev_file}")
print(f"📁 Test: {len(test_sentences)} ➝ {test_file}")

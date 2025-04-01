import os
import random

# Paths for input and output
DATA_DIR = os.path.join(os.path.dirname(__file__), "../")
without_entities_file = os.path.join(DATA_DIR, "sentences_without_entities.txt")

# Existing dataset files
train_file = os.path.join(DATA_DIR, "train.txt")
dev_file = os.path.join(DATA_DIR, "dev.txt")
test_file = os.path.join(DATA_DIR, "test.txt")

# Function to read sentences from a CoNLL file
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
                
        # Add last sentence if the file doesn't end with a newline
        if current_sentence:
            sentences.append("\n".join(current_sentence))
    
    return sentences

# Load sentences without entities
without_entities = read_sentences(without_entities_file)
random.shuffle(without_entities)

total = len(without_entities)
train_split = int(total * 0.5)
dev_split = int(total * 0.25)

train_without = without_entities[:train_split]
dev_without = without_entities[train_split:train_split + dev_split]
test_without = without_entities[train_split + dev_split:]

# Helper function to append sentences to files
def append_sentences(file_path, sentences):
    with open(file_path, "a", encoding="utf-8") as outfile:
        outfile.write("\n\n".join(sentences) + "\n\n")

# Append to existing datasets
append_sentences(train_file, train_without)
append_sentences(dev_file, dev_without)
append_sentences(test_file, test_without)

print(f"✅ Added sentences without entities to the datasets:")
print(f"   ➡️ Train: +{len(train_without)} sentences")
print(f"   ➡️ Dev: +{len(dev_without)} sentences")
print(f"   ➡️ Test: +{len(test_without)} sentences")

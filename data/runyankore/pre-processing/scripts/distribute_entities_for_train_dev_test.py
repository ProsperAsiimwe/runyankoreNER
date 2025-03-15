import os
import random

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
                    sentences.append("\n".join(current_sentence))
                    current_sentence = []
            else:
                current_sentence.append(line)
        # Add the last sentence if the file doesn't end with a newline
        if current_sentence:
            sentences.append("\n".join(current_sentence))
    
    return sentences

# Function to categorize sentences based on the presence of entity labels
def categorize_sentences(sentences, label):
    return [sentence for sentence in sentences if label in sentence]

# Read all sentences with entities
all_sentences = read_sentences(input_file)
print(f"🔍 Total sentences with entities: {len(all_sentences)}")

# Track already added sentences to avoid duplicates
added_sentences = set()
train_sentences, dev_sentences, test_sentences = [], [], []

# Process each entity label
for label in entity_labels:
    label_sentences = categorize_sentences(all_sentences, label)
    random.shuffle(label_sentences)  # Shuffle to randomize distribution

    # Remove duplicates
    label_sentences = [s for s in label_sentences if s not in added_sentences]
    
    total = len(label_sentences)
    train_split = int(total * 0.5)
    dev_split = int(total * 0.25)
    
    # Assign sentences
    train_sentences.extend(label_sentences[:train_split])
    dev_sentences.extend(label_sentences[train_split:train_split + dev_split])
    test_sentences.extend(label_sentences[train_split + dev_split:])
    
    # Track the sentences that have already been added
    added_sentences.update(label_sentences)

    print(f"✅ Processed label: {label}")
    print(f"   ➡️ Train: {train_split}, Dev: {dev_split}, Test: {total - train_split - dev_split}")

# Save the sentences to respective files
with open(train_file, "w", encoding="utf-8") as outfile:
    outfile.write("\n\n".join(train_sentences) + "\n\n")

with open(dev_file, "w", encoding="utf-8") as outfile:
    outfile.write("\n\n".join(dev_sentences) + "\n\n")

with open(test_file, "w", encoding="utf-8") as outfile:
    outfile.write("\n\n".join(test_sentences) + "\n\n")

print(f"✅ Splitting complete!")
print(f"📁 Train sentences saved to: {train_file}")
print(f"📁 Dev sentences saved to: {dev_file}")
print(f"📁 Test sentences saved to: {test_file}")

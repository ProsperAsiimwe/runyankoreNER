import os
from collections import Counter

# Set input file path
DATA_DIR = os.path.join(os.path.dirname(__file__), "../SALT/")
train_file = os.path.join(DATA_DIR, "train.txt")
dev_file = os.path.join(DATA_DIR, "dev.txt")
test_file = os.path.join(DATA_DIR, "test.txt")

def count_tokens_sentences_and_labels(file_path):
    token_count = 0
    sentence_count = 0
    label_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_sentence = False

    for line in lines:
        line = line.strip()

        # Blank line → sentence boundary
        if line == "":
            in_sentence = False
            continue

        parts = line.split()
        # Expect token + label or more columns; last column is label
        if len(parts) >= 2:
            token = parts[0]
            label = parts[-1]  # safely assume last column is label

            token_count += 1
            label_counts[label] += 1

            if not in_sentence:
                sentence_count += 1
                in_sentence = True

    return token_count, sentence_count, label_counts


# Example usage
if __name__ == "__main__":
    dataset_paths = {
        "Train": train_file,
        "Dev": dev_file,
        "Test": test_file
    }

    for split_name, path in dataset_paths.items():
        if os.path.exists(path):
            tokens, sentences, label_counts = count_tokens_sentences_and_labels(path)

            print(f"{split_name} set:")
            print(f"  Total tokens: {tokens}")
            print(f"  Total sentences: {sentences}")
            print(f"  Label counts:")
            for label in sorted(label_counts):
                print(f"    {label}: {label_counts[label]}")
            print()

        else:
            print(f"{split_name} set not found: {path}")

import os

# Set input file path
DATA_DIR = os.path.join(os.path.dirname(__file__), "../runyankore_multilingual_harvard_dataset/")
train_file = os.path.join(DATA_DIR, "train.txt") 
dev_file = os.path.join(DATA_DIR, "dev.txt") 
test_file = os.path.join(DATA_DIR, "test.txt") 

def count_tokens_and_sentences(file_path):
    token_count = 0
    sentence_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_sentence = False

    for line in lines:
        line = line.strip()

        if line == "":
            in_sentence = False
            continue

        # Each non-blank line should contain at least a token and a label
        parts = line.split()
        if len(parts) >= 1:
            token_count += 1
            if not in_sentence:
                sentence_count += 1
                in_sentence = True

    return token_count, sentence_count

# Example usage
if __name__ == "__main__":
    dataset_paths = {
        "Train": train_file,
        "Dev": dev_file,
        "Test": test_file
    }

    for split_name, path in dataset_paths.items():
        if os.path.exists(path):
            tokens, sentences = count_tokens_and_sentences(path)
            print(f"{split_name} set:")
            print(f"  Total tokens: {tokens}")
            print(f"  Total sentences: {sentences}\n")
        else:
            print(f"{split_name} set not found: {path}")

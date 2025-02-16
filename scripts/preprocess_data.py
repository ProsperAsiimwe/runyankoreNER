import os

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/fresh/")
LABELS_FILE = os.path.join(DATA_DIR, "labels.txt")

TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
DEV_FILE = os.path.join(DATA_DIR, "dev.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")

def load_labels(file_path):
    """Load valid labels from labels.txt"""
    with open(file_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())

def preprocess_file(file_path, valid_labels):
    """Cleans the dataset by removing unlabeled tokens and invalid labels."""
    cleaned_lines = []
    removed_lines = 0
    converted_labels = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            parts = line.strip().split()

            # Skip empty lines
            if not line.strip():
                removed_lines += 1
                continue

            # Ensure exactly two columns (word + label)
            if len(parts) != 2:
                print(f"🚨 Line {i} removed (incorrect format): {line.strip()}")
                removed_lines += 1
                continue

            token, label = parts

            # Remove lines where the token has no label (like the '200' case)
            if label == "":
                print(f"🚨 Line {i} removed (missing label): {line.strip()}")
                removed_lines += 1
                continue

            # Convert label '0' to 'O'
            if label == "0":
                label = "O"
                converted_labels += 1

            # Remove lines if the label is not in labels.txt
            if label not in valid_labels:
                print(f"❌ Line {i} removed (invalid label: {label}): {line.strip()}")
                removed_lines += 1
                continue

            cleaned_lines.append(f"{token} {label}\n")

    # Save cleaned file
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(cleaned_lines)

    # Summary
    print(f"\n✅ {file_path} cleaned successfully!")
    print(f"   - Removed {removed_lines} lines (empty, missing label, or invalid).")
    print(f"   - Converted {converted_labels} occurrences of '0' to 'O'.\n")

if __name__ == "__main__":
    valid_labels = load_labels(LABELS_FILE)

    # Process train.txt, dev.txt and test.txt
    preprocess_file(TRAIN_FILE, valid_labels)
    preprocess_file(DEV_FILE, valid_labels)
    preprocess_file(TEST_FILE, valid_labels)

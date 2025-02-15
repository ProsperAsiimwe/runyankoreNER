import os
from collections import defaultdict

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/runyankore/train.txt")

def find_label_lines(file_path):
    label_lines = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            parts = line.strip().split()

            # Ensure the line contains exactly two columns (token + label)
            if len(parts) == 2:
                token, label = parts
                label_lines[label].append(i)
            elif len(parts) == 1:  # Handle cases where label is missing
                label_lines["MISSING_LABEL"].append(i)

    return label_lines

if __name__ == "__main__":
    label_lines = find_label_lines(DATA_DIR)

    print("\n🚀 **Label Occurrences in train.txt**\n")
    for label, lines in label_lines.items():
        print(f"{label}: {lines}")

import os

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/runyankore/train.txt")

def find_missing_labels(file_path):
    missing_label_lines = []

    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            parts = line.strip().split()

            # Check if the line contains only one column (missing label)
            if len(parts) == 1:
                missing_label_lines.append(i)

    return missing_label_lines

if __name__ == "__main__":
    missing_labels = find_missing_labels(DATA_DIR)

    if missing_labels:
        print("\n🚨 **Lines with Missing Labels**:\n")
        print(missing_labels)
    else:
        print("\n✅ No missing labels found! Your dataset is clean.\n")

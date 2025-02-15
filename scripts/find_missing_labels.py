import os

# Define the path to the dataset (relative to /scripts/)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/runyankore/train.txt")

def find_missing_labels(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    missing_label_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line and len(line.split()) == 1:  # If a line has only one column (token but no label)
            missing_label_lines.append((i + 1, line))  # Store line number and token

    if missing_label_lines:
        print("\n🚨 Found tokens without labels:\n")
        for line_num, token in missing_label_lines:
            print(f"Line {line_num}: {token}")
    else:
        print("\n✅ No missing labels found! Your dataset is clean.\n")

# Run the script
if __name__ == "__main__":
    find_missing_labels(DATA_DIR)

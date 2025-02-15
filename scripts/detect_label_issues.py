import os

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/runyankore/train.txt")
LABELS_FILE = os.path.join(os.path.dirname(__file__), "../data/runyankore/labels.txt")

def load_labels(file_path):
    """Load valid labels from labels.txt"""
    with open(file_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())

def find_label_issues(file_path, labels):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    missing_label_lines = []
    blank_lines = []
    extra_space_lines = []
    unknown_label_lines = []

    for i, line in enumerate(lines):
        line = line.rstrip()  # Remove trailing spaces
        
        # Check for completely blank lines
        if line == "":
            blank_lines.append(i + 1)
            continue

        parts = line.split()
        
        # Check if a token is missing a label (only one column)
        if len(parts) == 1:
            missing_label_lines.append((i + 1, parts[0]))
            continue
        
        # Check if label is unknown
        token, label = parts[0], parts[1]
        if label not in labels:
            unknown_label_lines.append((i + 1, token, label))

        # Check for extra spaces before labels
        if line.endswith(" "):
            extra_space_lines.append(i + 1)

    return missing_label_lines, blank_lines, extra_space_lines, unknown_label_lines

if __name__ == "__main__":
    labels = load_labels(LABELS_FILE)
    missing_labels, blank_lines, extra_spaces, unknown_labels = find_label_issues(DATA_DIR, labels)

    if missing_labels:
        print("\n🚨 Found tokens without labels:\n")
        for line_num, token in missing_labels:
            print(f"Line {line_num}: {token}")

    if blank_lines:
        print("\n🚨 Found completely blank lines:\n")
        for line_num in blank_lines:
            print(f"Line {line_num}")

    if extra_spaces:
        print("\n🚨 Found lines with extra spaces before labels:\n")
        for line_num in extra_spaces:
            print(f"Line {line_num}")

    if unknown_labels:
        print("\n🚨 Found unknown labels that are not in labels.txt:\n")
        for line_num, token, label in unknown_labels:
            print(f"Line {line_num}: {token} -> {label}")

    if not (missing_labels or blank_lines or extra_spaces or unknown_labels):
        print("\n✅ No label issues found! Your dataset is clean.\n")

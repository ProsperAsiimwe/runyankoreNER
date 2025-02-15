import os
import re

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
    extra_column_lines = []
    non_printable_lines = []

    for i, line in enumerate(lines):
        original_line = line.rstrip()  # Remove trailing spaces

        # Check for completely blank lines
        if line.strip() == "":
            blank_lines.append(i + 1)
            continue

        parts = line.split()

        # Check if a token is missing a label (only one column)
        if len(parts) == 1:
            missing_label_lines.append((i + 1, parts[0]))
            continue

        # Check if the line has more than two columns (extra tokens)
        if len(parts) > 2:
            extra_column_lines.append((i + 1, original_line))
            continue

        # Extract token and label
        token, label = parts[0], parts[1]

        # Check if label is unknown
        if label not in labels:
            unknown_label_lines.append((i + 1, token, label))

        # Check for extra spaces before labels
        if original_line.endswith(" "):
            extra_space_lines.append(i + 1)

        # Check for non-printable characters (excluding normal whitespace)
        if re.search(r'[^\x09\x0A\x0D\x20-\x7E]', original_line):  # Allows tabs, newlines, carriage returns
            non_printable_lines.append((i + 1, original_line))

    return missing_label_lines, blank_lines, extra_space_lines, unknown_label_lines, extra_column_lines, non_printable_lines

if __name__ == "__main__":
    labels = load_labels(LABELS_FILE)
    missing_labels, blank_lines, extra_spaces, unknown_labels, extra_columns, non_printable = find_label_issues(DATA_DIR, labels)

    # Only print detected issues
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

    if extra_columns:
        print("\n🚨 Found lines with more than two columns (extra tokens detected):\n")
        for line_num, line in extra_columns:
            print(f"Line {line_num}: {line.strip()}")

    if non_printable:
        print("\n🚨 Found non-printable characters in these lines:\n")
        for line_num, line in non_printable:
            print(f"Line {line_num}: {line.strip()}")

    if not (missing_labels or blank_lines or extra_spaces or unknown_labels or extra_columns or non_printable):
        print("\n✅ No label issues found! Your dataset is clean.\n")

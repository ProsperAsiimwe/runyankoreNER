import os

# Define file paths
DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/runyankore/train.txt")

def replace_missing_label(file_path):
    """Finds and replaces the missing label with 'X'."""
    updated_lines = []
    modified_line = None

    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            parts = line.strip().split()

            # Check for a line with only one column (token without a label)
            if len(parts) == 1:
                modified_line = i  # Save the line number
                updated_lines.append(f"{parts[0]} X\n")  # Replace missing label with 'X'
            else:
                updated_lines.append(line)  # Keep original line

    # Save the modified file
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(updated_lines)

    # Print result
    if modified_line:
        print(f"\n🚨 Replaced missing label at **Line {modified_line}** with 'X'. Please review manually!\n")
    else:
        print("\n✅ No missing labels found. No changes made.\n")

if __name__ == "__main__":
    replace_missing_label(DATA_FILE)

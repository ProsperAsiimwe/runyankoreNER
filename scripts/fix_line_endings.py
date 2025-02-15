import os

# Define paths
DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/runyankore/train.txt")

def fix_line_endings(file_path):
    """Converts Windows line endings (CRLF) to Unix (LF) and removes trailing spaces."""
    with open(file_path, "rb") as file:
        content = file.read()

    # Convert CRLF (`\r\n`) to LF (`\n`) and remove trailing spaces
    fixed_content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n").strip()

    with open(file_path, "wb") as file:
        file.write(fixed_content)

if __name__ == "__main__":
    fix_line_endings(DATA_FILE)
    print("\n✅ File successfully converted to Unix format (LF)!\n")

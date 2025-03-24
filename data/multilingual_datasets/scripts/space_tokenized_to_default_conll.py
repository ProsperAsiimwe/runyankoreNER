import os

# Define your input/output file paths
base_dir = "../SPACE_TOKENIZED_RUNYANKORE/"
input_path = os.path.join(base_dir, "Multilingual_Parallel_Corpus.txt")
output_path = os.path.join(base_dir, "Multilingual_Parallel_Corpus.conll")

def convert_to_conll(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            for token in tokens:
                outfile.write(f"{token} O\n")
            outfile.write("\n")  # separate sentences with a blank line

    print(f"✅ Converted CoNLL file saved to: {output_file}")

# Run the conversion
convert_to_conll(input_path, output_path)

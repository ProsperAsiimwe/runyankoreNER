import os

# Paths to input and output files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../")
input_file = os.path.join(DATA_DIR, "verified_combined_boi_runyankore_sentences_v2.txt")
output_with_entities = os.path.join(DATA_DIR, "sentences_with_entities.txt")
output_without_entities = os.path.join(DATA_DIR, "sentences_without_entities.txt")

# NER entity labels to check
entity_labels = {"B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE"}

print(f"Processing CoNLL file: {input_file}")

# Buffers for storing sentences
sentences_with_entities = []
sentences_without_entities = []
current_sentence = []
has_entity = False  # Flag to track if sentence has an entity

# Read and process the input CoNLL file
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        
        if not line:  # Sentence separator
            if current_sentence:  # Ensure not to write empty sentences
                if has_entity:
                    sentences_with_entities.append("\n".join(current_sentence) + "\n\n")
                else:
                    sentences_without_entities.append("\n".join(current_sentence) + "\n\n")
            current_sentence = []
            has_entity = False  # Reset flag for next sentence
            continue  # Skip further processing for empty lines

        # Ensure the line contains both word and label
        if " " not in line:
            print(f"Skipping malformed line: {line}")  # Debugging
            continue

        # Extract word and label
        word, label = line.rsplit(" ", 1)  # Split by last space

        # Add to sentence buffer
        current_sentence.append(line)

        # Check if the label is an entity
        if label in entity_labels:
            has_entity = True  # Mark if entity is found

# Write sentences with entities
with open(output_with_entities, "w", encoding="utf-8") as outfile:
    outfile.writelines(sentences_with_entities)

# Write sentences without entities
with open(output_without_entities, "w", encoding="utf-8") as outfile:
    outfile.writelines(sentences_without_entities)

print(f"Sentences with entities saved to: {output_with_entities}")
print(f"Sentences without entities saved to: {output_without_entities}")

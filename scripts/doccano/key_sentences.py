import json
import os
from transformers import pipeline
from tqdm import tqdm

data_base_path = './DATA/'

# Load English NER Model
print("🚀 Loading English NER Model for Sentence Filtering...")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")

# Input dataset file
input_jsonl_file = os.path.join(data_base_path, 'CLEAN_DATA_SALT_SPACE_TOKENIZED_DOCCANO/test_salt_tokenized-eng-nyn-doccano.jsonl')  # Update with actual file path
filtered_output_file = os.path.join(data_base_path, 'RELEVANT_SENTENCES/filtered_runyankore_sentences.txt')

# Define Entity Categories (MasakhaNER)
target_entities = {"PER", "LOC", "ORG", "DATE"}

# Initialize Counters
total_sentences = 0
sentences_with_entities = 0

# Process Each Sentence
print(f"🔍 Searching for key sentences in: {input_jsonl_file}")

with open(input_jsonl_file, "r", encoding="utf-8") as infile, open(filtered_output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Processing Sentences"):
        total_sentences += 1
        data = json.loads(line)

        runyankore_text = data["text"]
        english_translation = data["metadata"]["translation"]

        # Perform NER on the English translation
        ner_results = ner_pipeline(english_translation)

        # Extract recognized entity types
        found_entities = {entity["entity"] for entity in ner_results}

        # Check if sentence contains at least one target entity
        if target_entities.intersection(found_entities):
            sentences_with_entities += 1
            outfile.write(json.dumps(data) + "\n")

# Summary
print("\n✅ Processing Complete!")
print(f"📌 Total Sentences Processed: {total_sentences}")
print(f"📍 Sentences Containing Entities: {sentences_with_entities}")
print(f"💾 Filtered Sentences Saved to: {filtered_output_file}")
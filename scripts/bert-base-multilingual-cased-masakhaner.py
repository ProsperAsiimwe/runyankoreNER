import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/runyankore/")

# Specify the path to MasakhaNER model
masakhaner_model = "Davlan/bert-base-multilingual-cased-masakhaner"

print(f"🚀 Loading MasakhaNER model: {masakhaner_model}")

# Load the MasakhaNER model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(masakhaner_model)
model = AutoModelForTokenClassification.from_pretrained(masakhaner_model)

# Load NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Input and output files
input_file = os.path.join(DATA_DIR, "combined_runyankore_sentences.txt")
output_file = os.path.join(DATA_DIR, "runyankore_masakhaner_predictions.conll")

print("🚀 Performing zero-shot NER on Runyankore sentences using MasakhaNER...")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        sentence = line.strip()
        if not sentence:
            continue  # Skip empty lines
        
        # Run the model for NER prediction
        ner_results = ner_pipeline(sentence)

        # Write in CoNLL-03 format
        for word in sentence.split():
            ner_label = "O"  # Default label
            for entity in ner_results:
                if word in entity["word"]:
                    ner_label = entity["entity_group"]
                    break
            
            outfile.write(f"{word} {ner_label}\n")
        
        outfile.write("\n")  # Separate sentences by a newline

print(f"✅ Zero-shot NER predictions saved to: {output_file}")

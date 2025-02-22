from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/luganda/")

# Specify the path to your fine-tuned Luganda NER model
trained_model_path = os.path.join(MODEL_DIR, "luganda_xlmr")  # Update this path

print(f"🚀 Loading your fine-tuned Luganda NER model from: {trained_model_path}")

# Load the fine-tuned Luganda NER model
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForTokenClassification.from_pretrained(trained_model_path)

# Load NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Input and output files
input_file = os.path.join(DATA_DIR, "combined_runyankore_sentences.txt") 
output_file = os.path.join(DATA_DIR, "runyankore_ner_predictions.conll")

print("🚀 Performing zero-shot NER on Runyankore sentences using your fine-tuned Luganda model...")

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

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/multilingual_datasets/")

# Specify the path to your fine-tuned Runyankore NER model
trained_model_path = os.path.join(MODEL_DIR, "runyankore_xlmr")  # Update this path

print(f"🚀 Loading your fine-tuned Runyankore NER model from: {trained_model_path}")

# Load the fine-tuned Runyankore NER model
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForTokenClassification.from_pretrained(trained_model_path)

# Load NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Input and output files
input_file = os.path.join(DATA_DIR, "/SPACE_TOKENIZED_RUNYANKORE/Multilingual_Parallel_Corpus.txt") 
output_file = os.path.join(DATA_DIR, "/SPACE_TOKENIZED_RUNYANKORE/Multilingual_Parallel_Corpus.conll")

print("🚀 Performing zero-shot NER on Runyankore sentences using your fine-tuned Runyankore model...")

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



# Aggregation Strategy = "Simple"

# In the script, you set aggregation_strategy="simple", which groups tokens into whole entities but does not assign BIO prefixes (B-, I-, O).
# Instead, the output directly provides the entity type (ORG, LOC, etc.) or O for non-entity words.
# How Tokenization Affects Labeling

# AutoTokenizer tokenizes words into subword pieces.
# When an entity spans multiple tokens, the model returns one entity label for the entire span instead of marking B- and I- labels.
# Hugging Face Pipelines Do Not Assign BIO Tags by Default

# The pipeline("ner") function is designed for high-level entity recognition, not structured sequence labeling.
# To get BIO-formatted labels, you must manually process the output.

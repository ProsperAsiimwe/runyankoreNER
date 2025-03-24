import os
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

data_base_path = '../'  # Adjust as needed

# Step 1: Load the spaCy language model
nlp = English()

# Step 2: Customize the tokenizer to treat punctuation as separate tokens
def custom_tokenizer(nlp):
    infixes = list(nlp.Defaults.infixes)
    punctuation = [r'\.', r',', r'\?', r'!', r':', r';']
    infixes.extend(punctuation)
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

nlp.tokenizer = custom_tokenizer(nlp)

# Step 3: Tokenize a .txt file and save output to a new .txt file
def tokenize_txt_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            sentence = line.strip()
            if not sentence:
                continue

            # Tokenize the sentence
            doc = nlp(sentence)
            tokens = [token.text for token in doc]

            # Join tokens with a space and write to output
            tokenized_sentence = " ".join(tokens)
            outfile.write(tokenized_sentence + "\n")

# File paths
input_path = os.path.join(data_base_path, 'RAW_RUNYANKORE/Multilingual_Parallel_Corpus.txt')
output_path = os.path.join(data_base_path, 'SPACE_TOKENIZED_RUNYANKORE/Multilingual_Parallel_Corpus.txt')

# Run the tokenizer
tokenize_txt_file(input_path, output_path)

print(f"✅ Tokenized sentences saved to: {output_path}")

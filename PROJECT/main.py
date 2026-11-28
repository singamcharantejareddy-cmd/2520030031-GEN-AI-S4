
# Comparative Analysis of Multilingual Tokenization
# Project: Generative AI and AI Platform Engineering

import pandas as pd
import os
import time

from transformers import AutoTokenizer

# ----------------------------------
# STEP 1: Load Dataset
# ----------------------------------

print("=" * 60)
print("MULTILINGUAL TOKENIZATION ANALYSIS")
print("=" * 60)

dataset_path = "dataset/multilingual_tokenization_sample_dataset.csv"

if not os.path.exists(dataset_path):
    print("Dataset file not found!")
    exit()

data = pd.read_csv(dataset_path)

print("\nDataset Loaded Successfully!")
print("Total Sentences:", len(data))
print("Available Languages:")
print(["English", "Telugu", "Hindi", "Tamil", "Kannada"])


# ----------------------------------
# STEP 2: Load Tokenizers
# ----------------------------------

print("\nLoading Tokenizers...")

tokenizers = {
    "BPE": "gpt2",
    "WordPiece": "bert-base-multilingual-cased",
    "SentencePiece": "google/mt5-small"
}

loaded_tokenizers = {}

for name, model in tokenizers.items():

    print("Loading", name)

    loaded_tokenizers[name] = AutoTokenizer.from_pretrained(
        model
    )

print("\nAll Tokenizers Loaded Successfully!")


# ----------------------------------
# STEP 3: Select Languages
# ----------------------------------

languages = [
    "english",
    "telugu",
    "hindi",
    "tamil",
    "kannada"
]


# ----------------------------------
# STEP 4: Tokenization Function
# ----------------------------------

def count_tokens(text, tokenizer):

    tokens = tokenizer.tokenize(str(text))

    return len(tokens), tokens


# ----------------------------------
# STEP 5: Compare Token Counts
# ----------------------------------

results = []

print("\nStarting Tokenization Analysis...\n")

for index, row in data.iterrows():

    for language in languages:

        text = row[language]

        for tokenizer_name, tokenizer in loaded_tokenizers.items():

            start_time = time.perf_counter()

            count, tokens = count_tokens(
                text,
                tokenizer
            )

            end_time = time.perf_counter()

            processing_time = end_time - start_time

            results.append({
                "Sentence_ID": row["sentence_id"],
                "Language": language,
                "Tokenizer": tokenizer_name,
                "Text": text,
                "Token_Count": count,
                "Tokens": " | ".join(tokens),
                "Processing_Time": processing_time
            })


# ----------------------------------
# STEP 6: Display Results
# ----------------------------------

result_df = pd.DataFrame(results)

print("\nTOKENIZATION RESULTS")
print(result_df.head(15).to_string(index=False))


# ----------------------------------
# STEP 7: Save Results
# ----------------------------------

os.makedirs("results", exist_ok=True)

result_df.to_csv(
    "results/tokenization_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAnalysis Completed Successfully!")
print("Results saved in results/tokenization_results.csv")

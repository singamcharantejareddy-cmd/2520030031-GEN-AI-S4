
# WEEK 5: TOKENIZATION AND TOKEN ECONOMICS
# English vs Telugu vs Hindi
# API Cost Calculation and Budgeting

from transformers import AutoTokenizer
import csv
import os

# --------------------------------------------------
# PART 1: ENGLISH, TELUGU AND HINDI TOKEN COMPARISON
# --------------------------------------------------

sentences = {
    "English": "Artificial Intelligence helps students learn new technologies.",

    "Telugu": "కృత్రిమ మేధస్సు విద్యార్థులకు కొత్త సాంకేతికతలను నేర్చుకోవడంలో సహాయపడుతుంది.",

    "Hindi": "कृत्रिम बुद्धिमत्ता छात्रों को नई तकनीक सीखने में मदद करती है।"
}

tokenizer_models = {
    "BPE (GPT-2)": {
        "model": "openai-community/gpt2",
        "use_fast": True
    },

    "WordPiece (Multilingual BERT)": {
        "model": "google-bert/bert-base-multilingual-cased",
        "use_fast": True
    },

    "SentencePiece (mT5)": {
        "model": "google/mt5-small",
        "use_fast": False
    }
}

print("\n======================================")
print("WEEK 5: TOKENIZATION COMPARISON")
print("======================================")

# Load tokenizers
loaded_tokenizers = {}

for name, config in tokenizer_models.items():

    print("\nLoading:", name)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        use_fast=config["use_fast"]
    )

    loaded_tokenizers[name] = tokenizer

    print("Loaded successfully!")

# Tokenize sentences
token_results = []

for language, sentence in sentences.items():

    print("\n--------------------------------------")
    print("Language:", language)
    print("Sentence:", sentence)

    word_count = len(sentence.split())

    for tokenizer_name, tokenizer in loaded_tokenizers.items():

        token_ids = tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        token_count = len(token_ids)

        token_ratio = (
            token_count / word_count
            if word_count > 0 else 0
        )

        print("\nTokenizer:", tokenizer_name)
        print("Tokens:", tokens)
        print("Word Count:", word_count)
        print("Token Count:", token_count)
        print("Token Ratio:", round(token_ratio, 3))

        token_results.append({
            "Language": language,
            "Tokenizer": tokenizer_name,
            "Word_Count": word_count,
            "Token_Count": token_count,
            "Token_Ratio": round(token_ratio, 3)
        })

# Save tokenizer comparison to CSV
os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/week5_token_comparison.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "Language",
        "Tokenizer",
        "Word_Count",
        "Token_Count",
        "Token_Ratio"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(token_results)

print("\nToken comparison saved successfully!")

# --------------------------------------------------
# PART 2: TOKEN ECONOMICS AND COST CALCULATION
# --------------------------------------------------

print("\n======================================")
print("LLM API COST CALCULATOR")
print("======================================")

# Example API pricing (hypothetical USD rates)
INPUT_RATE = 0.50
OUTPUT_RATE = 2.00

# Student chatbot usage assumptions
students = 1000
requests_per_student = 10

input_tokens_per_request = 1500
output_tokens_per_request = 500

days_per_month = 30

# Step 1: Total requests per day
total_requests = students * requests_per_student

# Step 2: Calculate total input tokens
total_input_tokens = (
    total_requests * input_tokens_per_request
)

# Step 3: Calculate total output tokens
total_output_tokens = (
    total_requests * output_tokens_per_request
)

# Step 4: Calculate input cost
input_cost = (
    total_input_tokens * INPUT_RATE / 1_000_000
)

# Step 5: Calculate output cost
output_cost = (
    total_output_tokens * OUTPUT_RATE / 1_000_000
)

# Step 6: Calculate total daily cost
daily_cost = input_cost + output_cost

# Step 7: Calculate monthly cost
monthly_cost = daily_cost * days_per_month

# Display results
print("\nCHATBOT USAGE DETAILS")

print("Number of Students:", students)
print("Requests per Student:", requests_per_student)
print("Total Requests per Day:", total_requests)

print("\nTOKEN DETAILS")

print("Input Tokens per Request:", input_tokens_per_request)
print("Output Tokens per Request:", output_tokens_per_request)

print("Total Daily Input Tokens:", total_input_tokens)
print("Total Daily Output Tokens:", total_output_tokens)

print("\nCOST DETAILS")

print("Input Rate per Million Tokens: $", INPUT_RATE)
print("Output Rate per Million Tokens: $", OUTPUT_RATE)

print("Daily Input Cost: $", round(input_cost, 4))
print("Daily Output Cost: $", round(output_cost, 4))

print("Total Daily Cost: $", round(daily_cost, 4))
print("Total Monthly Cost: $", round(monthly_cost, 2))

# --------------------------------------------------
# PART 3: CONTEXT WINDOW CALCULATION
# --------------------------------------------------

print("\n======================================")
print("CONTEXT WINDOW CALCULATION")
print("======================================")

context_window = 128000

system_tokens = 2000
conversation_tokens = 10000
user_input_tokens = 1500

used_context = (
    system_tokens
    + conversation_tokens
    + user_input_tokens
)

remaining_context = context_window - used_context

print("Maximum Context Window:", context_window)
print("System Tokens:", system_tokens)
print("Conversation Tokens:", conversation_tokens)
print("Current User Input Tokens:", user_input_tokens)

print("Total Used Context:", used_context)
print("Remaining Context:", remaining_context)

# --------------------------------------------------
# PART 4: FINAL SUMMARY
# --------------------------------------------------

print("\n======================================")
print("WEEK 5 EXPERIMENT COMPLETED")
print("======================================")

print("1. Language token comparison completed.")
print("2. Input and output token costs calculated.")
print("3. Daily and monthly budgets calculated.")
print("4. Context window usage calculated.")

print("\nGenerated File: outputs/week5_token_comparison.csv")

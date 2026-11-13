# Calculate the estimated cost of LLM requests

PRICING = {
    "gemini-2.5-flash": {
        "input_per_million": 0.30,
        "output_per_million": 2.50
    }
}


def calculate_cost(model, input_tokens, output_tokens):

    if model.lower() in ["llama3.2", "ollama"]:
        return 0.0

    rates = PRICING.get(model)

    if rates is None:
        return 0.0

    input_cost = (
        input_tokens / 1000000
    ) * rates["input_per_million"]

    output_cost = (
        output_tokens / 1000000
    ) * rates["output_per_million"]

    return round(input_cost + output_cost, 8)
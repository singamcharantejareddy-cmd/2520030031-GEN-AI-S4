from abc import ABC, abstractmethod
import os
import time
import requests

from google import genai
from google.genai import types

from cost_calculator import calculate_cost


# Common interface for all LLM providers
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# Gemini cloud provider
class GeminiProvider(LLMProvider):

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key or api_key.startswith("PASTE_"):
            raise ValueError("Please add your Gemini API key in .env")

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024
            )
        )

        usage = response.usage_metadata

        return {
            "text": response.text or "",
            "input_tokens": getattr(
                usage, "prompt_token_count", 0
            ) or 0,
            "output_tokens": getattr(
                usage, "candidates_token_count", 0
            ) or 0
        }


# Local Ollama provider
class OllamaProvider(LLMProvider):

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")

        self.url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate"
        )

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return {
            "text": data.get("response", ""),
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0)
        }


# Common LLM client
class LLMClient:

    def __init__(self, provider, logger=None, max_retries=3):
        self.provider = provider
        self.logger = logger
        self.max_retries = max_retries

    def generate(self, prompt):

        last_error = None

        for attempt in range(self.max_retries):

            try:
                start = time.perf_counter()

                result = self.provider.generate(prompt)

                latency = round(
                    (time.perf_counter() - start) * 1000,
                    2
                )

                model = getattr(
                    self.provider,
                    "model",
                    type(self.provider).__name__
                )

                input_tokens = int(result.get("input_tokens", 0))
                output_tokens = int(result.get("output_tokens", 0))

                cost = calculate_cost(
                    model,
                    input_tokens,
                    output_tokens
                )

                metrics = {
                    "provider": type(self.provider).__name__,
                    "model": model,
                    "latency_ms": latency,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "estimated_cost_usd": cost,
                    "success": True
                }

                if self.logger:
                    self.logger.log_call(metrics)

                return {
                    "text": result["text"],
                    "metrics": metrics
                }

            except Exception as error:

                last_error = error

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        if self.logger:
            self.logger.log_call({
                "provider": type(self.provider).__name__,
                "success": False,
                "error": str(last_error)
            })

        raise RuntimeError(
            f"Request failed after {self.max_retries} attempts: {last_error}"
        )
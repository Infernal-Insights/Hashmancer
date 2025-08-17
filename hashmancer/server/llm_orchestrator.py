import os
import json
import logging
import re

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover - transformers optional
    AutoModelForCausalLM = None
    AutoTokenizer = None
    logging.warning("transformers unavailable: %s", e)


class LLMOrchestrator:
    """Lightweight interface around a local language model."""

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers library is required")
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH", "distilgpt2")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model.to(device)
        except Exception as e:  # pragma: no cover - optional component
            logging.warning("Failed to load LLM model %s: %s", self.model_path, e)
            self.model = None
            self.tokenizer = None

    def _generate(self, prompt: str) -> str:
        if not self.model or not self.tokenizer:
            return ""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_new_tokens=16)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def suggest_batch_size(self, batch_info: dict) -> tuple[int, int]:
        """Return a (start, end) tuple for the given batch."""
        prompt = (
            "Suggest start and end numbers for this batch.\n" +
            json.dumps(batch_info) + "\nFormat: start,end"
        )
        reply = self._generate(prompt)
        match = re.search(r"(\d+)[,\s]+(\d+)", reply)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 1000

    def choose_job_stream(
        self, batch_info: dict, high_pending: int, low_pending: int
    ) -> str:
        """Return 'high' or 'low' for the preferred queue."""
        prompt = (
            "Choose the job queue for this batch.\n" +
            json.dumps(batch_info) +
            f"\nHigh pending: {high_pending} Low pending: {low_pending}\n" +
            "Respond with 'high' or 'low'."
        )
        reply = self._generate(prompt).lower()
        return "low" if "low" in reply else "high"


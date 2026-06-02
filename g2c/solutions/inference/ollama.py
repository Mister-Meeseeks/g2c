# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.inference.ollama pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from typing import Any

from g2c.inference.backend import InferenceResult


class _OllamaBackendImpl:  # patched onto OllamaBackend by apply()
    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        think: bool | None = None,
    ) -> InferenceResult:
        """Map params + response fields onto the Backend contract.

        `_post_generate` (a sibling method on OllamaBackend) owns the
        HTTP transport; this method is the contract mapping the
        Module-16 lesson is about.
        """
        if not isinstance(prompt, str) or len(prompt) == 0:
            raise ValueError("prompt must be a non-empty str")
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {max_new_tokens}"
            )

        options: dict[str, Any] = {
            "num_predict": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if top_k is not None:
            options["top_k"] = int(top_k)
        if top_p is not None:
            options["top_p"] = float(top_p)

        body: dict[str, Any] = {
            "model": self._model_id,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if think is not None:
            body["think"] = bool(think)

        data, latency_ms = self._post_generate(body)

        total_duration = data.get("total_duration")
        eval_duration = data.get("eval_duration")
        return InferenceResult(
            prompt=prompt,
            completion=data["response"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
            backend=self._info,
            metadata={
                "sampling": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                "server_total_duration_ms": (
                    float(total_duration) / 1e6 if total_duration is not None else None
                ),
                "server_eval_duration_ms": (
                    float(eval_duration) / 1e6 if eval_duration is not None else None
                ),
            },
        )

"""Provider interface and the structured completion result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CompletionResult:
    content: str
    stop_reason: str | None = None
    stop_details: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    native_config: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None


class Provider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> CompletionResult:
        """Run one prompt in the model's best native config; never raises."""
        ...

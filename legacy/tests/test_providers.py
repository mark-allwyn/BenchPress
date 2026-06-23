"""Tests for scripts/providers.py - sanitization, factory, dataclass, ABC."""

import os
import pytest
from scripts.providers import (
    sanitize_error,
    get_provider,
    ModelResponse,
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    GoogleProvider,
    OllamaProvider,
    CohereProvider,
    BedrockProvider,
)


# ── sanitize_error ──

class TestSanitizeError:
    def test_redacts_anthropic_key(self):
        msg = "Error with key sk-ant-api03-abc123xyz"
        result = sanitize_error(msg)
        assert "sk-ant-api" not in result
        assert "[REDACTED]" in result

    def test_redacts_openai_key(self):
        msg = "Error with key sk-proj-abc123xyz"
        result = sanitize_error(msg)
        assert "sk-proj-" not in result
        assert "[REDACTED]" in result

    def test_redacts_bearer_token(self):
        msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc"
        result = sanitize_error(msg)
        assert "eyJ" not in result
        assert "[REDACTED]" in result

    def test_redacts_google_key(self):
        msg = "key=AIzaSyABC123def456"
        result = sanitize_error(msg)
        assert "AIzaSy" not in result

    def test_redacts_xai_key(self):
        msg = "Error: xai-abc123xyz"
        result = sanitize_error(msg)
        assert "xai-" not in result

    def test_redacts_hf_token(self):
        msg = "Token: hf_AbCdEf123456"
        result = sanitize_error(msg)
        assert "hf_" not in result

    def test_redacts_aws_key(self):
        msg = "Key: AKIAIOSFODNN7EXAMPLE"
        result = sanitize_error(msg)
        assert "AKIA" not in result

    def test_preserves_non_sensitive(self):
        msg = "Connection timeout after 30s"
        assert sanitize_error(msg) == msg

    def test_redacts_url_key_param(self):
        msg = "https://api.example.com?key=AIzaSyABC123"
        result = sanitize_error(msg)
        assert "key=[REDACTED]" in result

    def test_redacts_x_api_key_header(self):
        msg = "x-api-key: sk-ant-api03-secret"
        result = sanitize_error(msg)
        assert "sk-ant-api" not in result


# ── ModelResponse dataclass ──

class TestModelResponse:
    def test_construction(self):
        r = ModelResponse(
            model="gpt-test", prompt_id="T01", content="Hello",
            latency_s=1.5, input_tokens=10, output_tokens=20,
        )
        assert r.model == "gpt-test"
        assert r.error is None

    def test_defaults(self):
        r = ModelResponse(model="m", prompt_id="p", content="c", latency_s=0.1)
        assert r.input_tokens is None
        assert r.output_tokens is None
        assert r.error is None

    def test_with_error(self):
        r = ModelResponse(model="m", prompt_id="p", content="", latency_s=0.0, error="timeout")
        assert r.error == "timeout"


# ── Provider ABC ──

class TestProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Provider()


# ── get_provider factory ──

class TestGetProvider:
    def test_ollama(self):
        cfg = {"provider": "ollama", "model": "llama3"}
        p = get_provider(cfg)
        assert isinstance(p, OllamaProvider)

    def test_ollama_custom_url(self):
        cfg = {"provider": "ollama", "model": "llama3", "base_url": "http://myhost:11434/v1"}
        p = get_provider(cfg)
        assert isinstance(p, OllamaProvider)

    def test_anthropic(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-key-value")
        cfg = {"provider": "anthropic", "model": "claude-test", "api_key_env": "TEST_KEY"}
        p = get_provider(cfg)
        assert isinstance(p, AnthropicProvider)

    def test_openai(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-key-value")
        cfg = {"provider": "openai", "model": "gpt-test", "api_key_env": "TEST_KEY"}
        p = get_provider(cfg)
        assert isinstance(p, OpenAIProvider)

    def test_openai_compatible(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-key-value")
        cfg = {"provider": "openai_compatible", "model": "custom", "api_key_env": "TEST_KEY", "base_url": "http://x"}
        p = get_provider(cfg)
        assert isinstance(p, OpenAIProvider)

    def test_google(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-key-value")
        cfg = {"provider": "google", "model": "gemini", "api_key_env": "TEST_KEY"}
        p = get_provider(cfg)
        assert isinstance(p, GoogleProvider)

    def test_cohere(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-key-value")
        cfg = {"provider": "cohere", "model": "command", "api_key_env": "TEST_KEY"}
        p = get_provider(cfg)
        assert isinstance(p, CohereProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider({"provider": "unknown_provider", "model": "x", "api_key_env": "none"})

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        cfg = {"provider": "anthropic", "model": "claude", "api_key_env": "NONEXISTENT_KEY"}
        with pytest.raises(ValueError, match="Set env var"):
            get_provider(cfg)

    def test_api_key_env_none(self):
        cfg = {"provider": "openai", "model": "gpt-test", "api_key_env": "none"}
        p = get_provider(cfg)
        assert isinstance(p, OpenAIProvider)

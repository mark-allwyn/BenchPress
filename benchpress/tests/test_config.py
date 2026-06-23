import textwrap

from benchpress.config import load_models


def test_load_models_returns_model_specs(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent("""
        models:
          claude-opus-4.8:
            provider: anthropic
            model: claude-opus-4-8
            company: Anthropic
            type: closed
            launch_date: "2026-05-28"
            api_key_env: ANTHROPIC_API_KEY
            params:
              max_tokens: 4096
        judges:
          ignored: true
    """))

    models = load_models(cfg)

    assert "claude-opus-4.8" in models
    spec = models["claude-opus-4.8"]
    assert spec["provider"] == "anthropic"
    assert spec["model"] == "claude-opus-4-8"
    assert spec["company"] == "Anthropic"
    assert spec["launch_date"] == "2026-05-28"


def test_load_models_empty_when_no_models_section(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("judges:\n  x: 1\n")
    assert load_models(cfg) == {}

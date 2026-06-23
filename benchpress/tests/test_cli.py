import pytest

from benchpress import cli
from benchpress.providers.base import CompletionResult

SUBCOMMANDS = ["generate", "run", "score", "stats", "export", "audit"]


def test_parser_exposes_all_subcommands():
    parser = cli.build_parser()
    for cmd in SUBCOMMANDS:
        args = parser.parse_args([cmd, "--benchmark", "causal", "--model", "m"])
        assert args.command == cmd


def test_no_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main([])


def test_generate_runs_offline(capsys):
    rc = cli.main(["generate", "--seed", "7"])
    assert rc == 0
    assert "generate" in capsys.readouterr().out


def test_stats_offline_over_empty_dir(tmp_path, capsys):
    rc = cli.main(["stats", "--results-dir", str(tmp_path)])
    assert rc == 0
    assert "model" in capsys.readouterr().out  # header renders


class _FakeProvider:
    def complete(self, prompt):
        return CompletionResult(
            content="ADJUSTMENT_SET: {X}\nESTIMATE: 0.00\nIDENTIFIABLE: yes",
            stop_reason="end_turn",
        )


def test_run_score_stats_pipeline_with_mocked_provider(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_models", lambda path: {
        "fake": {"provider": "anthropic", "model": "claude-fable-5"}
    })
    monkeypatch.setattr(cli, "get_provider", lambda spec: _FakeProvider())

    rd = str(tmp_path)
    assert cli.main(["run", "--model", "fake", "--seed", "7", "--results-dir", rd]) == 0
    assert (tmp_path / "causal" / "fake.json").exists()
    assert cli.main(["score", "--model", "fake", "--seed", "7", "--results-dir", rd]) == 0
    assert cli.main(["stats", "--seed", "7", "--results-dir", rd]) == 0
    assert "fake" in capsys.readouterr().out

    out_file = tmp_path / "export.json"
    assert cli.main(["export", "--seed", "7", "--results-dir", rd, "--out", str(out_file)]) == 0
    payload = out_file.read_text()
    assert "ci95" in payload and "part_marginals" in payload

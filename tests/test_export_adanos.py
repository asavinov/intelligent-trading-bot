import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


REPO = Path(__file__).resolve().parents[1]


def response():
    return {
        "symbol": "BTC", "found": True, "mentions": 9999,
        "daily_trend": [
            {"date": "2026-01-04", "mentions": 30, "buzz_score": 100,
             "sentiment_score": -1, "bullish_pct": 0, "bearish_pct": 100},
            {"date": "2026-01-02", "mentions": 10, "buzz_score": None,
             "bullish_pct": None, "sentiment_score": 0.25, "ignored": "private"},
        ],
    }


@pytest.fixture
def cli(request):
    root = Path(os.environ.get("ADANOS_PROOF_DIR") or tempfile.mkdtemp(prefix="adanos-proof-"))
    work = root / request.node.name
    work.mkdir(parents=True)
    guard = work / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "def deny(event, args):\n"
        "    if event.startswith('socket.'):\n"
        "        raise RuntimeError('network disabled by offline proof')\n"
        "sys.addaudithook(deny)\n", encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(guard), "PYTHONDONTWRITEBYTECODE": "1"}
    commands = []

    def run(*args):
        command = [sys.executable, *map(str, args)]
        result = subprocess.run(command, cwd=REPO, env=env, capture_output=True,
                                text=True, timeout=60)
        commands.append({"command": command, "cwd": str(REPO),
                         "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        return result

    yield work, run
    hashes = {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in work.rglob("*") if p.is_file()}
    (work / "proof.json").write_text(json.dumps({
        "test": request.node.nodeid, "commands": commands, "sha256": hashes,
        "assertions": "See test source and pytest run log for result",
    }, indent=2), encoding="utf-8")


def export(work, run, payload, symbol="BTC"):
    source = work / "token.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    destination = work / "daily.csv"
    result = run("-m", "scripts.export_adanos", "--input-file", source,
                 "--output-file", destination, "--symbol", symbol)
    return result, destination


def test_export_sorted_nullable_subset(cli):
    work, run = cli
    result, destination = export(work, run, response(), "$btc")
    assert result.returncode == 0, result.stderr
    assert destination.read_text().splitlines() == [
        "timestamp,mentions,buzz_score,sentiment_score,bullish_pct,bearish_pct",
        "2026-01-02T00:00:00,10,,0.25,,",
        "2026-01-04T00:00:00,30,100,-1,0,100",
    ]
    assert "private" not in result.stdout + result.stderr


def test_zero_and_missing_optional_fields(cli):
    payload = {"symbol": "BTC", "found": True, "daily_trend": [
        {"date": "2024-02-29", "mentions": 0},
        {"date": "2024-03-01", "mentions": 1, "buzz_score": 0,
         "sentiment_score": 0, "bullish_pct": 100, "bearish_pct": 0},
    ]}
    result, destination = export(*cli, payload)
    assert result.returncode == 0, result.stderr
    assert destination.read_text().splitlines()[1:] == [
        "2024-02-29T00:00:00,0,,,,", "2024-03-01T00:00:00,1,0,0,100,0",
    ]


@pytest.mark.parametrize("payload", [
    None, [], "provider-secret", {"detail": "provider-secret"},
    {**response(), "symbol": "ETH"}, {**response(), "symbol": None},
    {**response(), "found": False}, {**response(), "found": 1},
    {k: v for k, v in response().items() if k != "found"},
    {**response(), "daily_trend": None}, {**response(), "daily_trend": []},
    {**response(), "daily_trend": {}},
    {k: v for k, v in response().items() if k != "daily_trend"},
    {**response(), "daily_trend": [None]},
    {**response(), "daily_trend": [response()["daily_trend"][0]] * 2},
])
def test_reject_response_without_output(cli, payload):
    result, destination = export(*cli, payload)
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "provider-secret" not in result.stdout + result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("field,value", [
    ("date", None), ("date", "2026-02-30"), ("date", "20260102"),
    ("date", "2026-01-02T00:00:00Z"), ("date", "2026-1-2"),
    ("date", "0000-01-01"), ("date", "2026-W01-1"),
    ("mentions", None), ("mentions", True), ("mentions", -1),
    ("mentions", 1.5), ("mentions", "10"),
    ("buzz_score", -0.1), ("buzz_score", 100.1), ("buzz_score", True),
    ("buzz_score", "12"), ("buzz_score", []), ("buzz_score", float("nan")),
    ("sentiment_score", -1.01), ("sentiment_score", 1.01),
    ("sentiment_score", float("inf")), ("sentiment_score", float("-inf")),
    ("sentiment_score", False), ("sentiment_score", {}),
    ("bullish_pct", -1), ("bullish_pct", 101), ("bullish_pct", 0.5),
    ("bullish_pct", True), ("bearish_pct", "50"), ("bearish_pct", 101),
    ("bearish_pct", -1), ("bearish_pct", 1.0),
])
def test_reject_invalid_daily_value_before_write(cli, field, value):
    payload = response()
    payload["daily_trend"][1][field] = value
    result, destination = export(*cli, payload)
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("field", ["date", "mentions"])
def test_reject_missing_required_daily_field(cli, field):
    payload = response()
    del payload["daily_trend"][0][field]
    result, destination = export(*cli, payload)
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("symbol", ["", "BTC-USDT", "BTC/USDT", "A" * 21, "$$BTC"])
def test_reject_invalid_expected_symbol(cli, symbol):
    result, destination = export(*cli, response(), symbol)
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize("content", [b'{"detail": "provider-secret",', b'\xff'])
def test_reject_unreadable_json_without_leaking_content(cli, content):
    work, run = cli
    source = work / "token.json"
    source.write_bytes(content)
    result = run("-m", "scripts.export_adanos", "--input-file", source,
                 "--output-file", work / "daily.csv", "--symbol", "BTC")
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "provider-secret" not in result.stdout + result.stderr
    assert not (work / "daily.csv").exists()


@pytest.mark.parametrize("kind", ["file", "symlink", "same-file", "missing-input", "missing-parent"])
def test_file_safety(cli, kind):
    work, run = cli
    source = work / "token.json"
    source.write_text(json.dumps(response()), encoding="utf-8")
    original = source.read_bytes()
    target = work / "daily.csv"
    if kind == "file":
        target.write_text("keep me", encoding="utf-8")
    elif kind == "symlink":
        target.symlink_to(source)
    elif kind == "same-file":
        target = source
    elif kind == "missing-input":
        source = work / "absent.json"
    else:
        target = work / "absent" / "daily.csv"
    result = run("-m", "scripts.export_adanos", "--input-file", source,
                 "--output-file", target, "--symbol", "BTC")
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert (work / "token.json").read_bytes() == original
    if kind == "file":
        assert target.read_text() == "keep me"
    elif kind in ("missing-input", "missing-parent"):
        assert not target.exists()


def test_full_native_merge_offline(cli):
    work, run = cli
    blocked = run("-c", "import socket; socket.socket()")
    assert blocked.returncode != 0
    assert "network disabled by offline proof" in blocked.stderr
    docs = (REPO / "docs/data-inputs.md").read_text(encoding="utf-8")
    config = json.loads(docs.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert config["merge_interpolate"] is False
    assert config["freq"] == "1D"
    assert config["train"] is True and config["train_length"] == 0
    assert config["features_horizon"] == 0
    for key in ("feature_sets", "label_sets", "train_feature_sets", "signal_sets", "output_sets"):
        assert config[key] == []
    config["data_folder"] = str(work)
    auxiliary = work / "ADANOS_BTC"
    auxiliary.mkdir()
    result, daily = export(auxiliary, run, response())
    assert result.returncode == 0, result.stderr
    prices = work / "BTCUSDT"
    prices.mkdir()
    (prices / "klines.csv").write_text(
        "timestamp,close\n2026-01-01T00:00:00,1\n2026-01-02T00:00:00,2\n"
        "2026-01-03T00:00:00,3\n2026-01-04T00:00:00,4\n2026-01-05T00:00:00,5\n",
        encoding="utf-8",
    )
    output = work / "ANALYSIS"
    output.mkdir()
    config_path = work / "merge.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    result = run("-m", "scripts.merge", "-c", config_path)
    assert result.returncode == 0, result.stdout + result.stderr
    with (output / "merged.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["timestamp"][:10] for row in rows] == [
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
    ]
    assert [row["close"] for row in rows] == ["1", "2", "3", "4"]
    assert [row["adanos_mentions"] for row in rows] == ["", "10.0", "", "30.0"]
    assert [row["adanos_sentiment_score"] for row in rows] == ["", "0.25", "", "-1.0"]
    assert [row["adanos_buzz_score"] for row in rows] == ["", "", "", "100.0"]
    assert [row["adanos_bullish_pct"] for row in rows] == ["", "", "", "0.0"]
    assert [row["adanos_bearish_pct"] for row in rows] == ["", "", "", "100.0"]
    assert set(rows[0]) == {"timestamp", "close", "adanos_mentions", "adanos_buzz_score",
                            "adanos_sentiment_score", "adanos_bullish_pct", "adanos_bearish_pct"}
    assert sorted(p.name for p in output.iterdir()) == ["merged.csv"]

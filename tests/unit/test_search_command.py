"""Tests for search command --format flag output."""
import csv
import io
import json

import pytest


def _make_fake_results():
    return [
        {"similarity": 0.82, "content": "Liver inflammation observed", "id": 1, "category": "adverse"},
        {"similarity": 0.71, "content": "Hepatic toxicity grade 2", "id": 2, "category": "adverse"},
    ]


def test_format_csv_output(capsys):
    """--format csv writes valid CSV with header to stdout."""
    from pgsemantic.commands.search import _results_to_csv

    results = _make_fake_results()
    _results_to_csv(results)
    captured = capsys.readouterr()

    reader = csv.DictReader(io.StringIO(captured.out))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["similarity"] == "0.82"
    assert rows[0]["content"] == "Liver inflammation observed"
    assert rows[0]["id"] == "1"


def test_format_json_output(capsys):
    """--format json writes a JSON array to stdout."""
    from pgsemantic.commands.search import _results_to_json

    results = _make_fake_results()
    _results_to_json(results)
    captured = capsys.readouterr()

    parsed = json.loads(captured.out)
    assert len(parsed) == 2
    assert parsed[0]["similarity"] == 0.82
    assert parsed[1]["content"] == "Hepatic toxicity grade 2"


def test_format_jsonl_output(capsys):
    """--format jsonl writes one JSON object per line to stdout."""
    from pgsemantic.commands.search import _results_to_jsonl

    results = _make_fake_results()
    _results_to_jsonl(results)
    captured = capsys.readouterr()

    lines = [ln for ln in captured.out.strip().split("\n") if ln]
    assert len(lines) == 2
    assert json.loads(lines[0])["similarity"] == 0.82
    assert json.loads(lines[1])["id"] == 2


def test_format_csv_empty_results(capsys):
    """_results_to_csv with empty list writes nothing (no header)."""
    from pgsemantic.commands.search import _results_to_csv

    _results_to_csv([])
    captured = capsys.readouterr()
    assert captured.out == ""


def test_format_csv_none_value(capsys):
    """_results_to_csv converts None values to empty string."""
    from pgsemantic.commands.search import _results_to_csv

    results = [{"id": 1, "content": None, "similarity": 0.5}]
    _results_to_csv(results)
    captured = capsys.readouterr()

    reader = csv.DictReader(io.StringIO(captured.out))
    rows = list(reader)
    assert rows[0]["content"] == ""


def test_format_json_empty_results(capsys):
    """_results_to_json with empty list writes '[]'."""
    from pgsemantic.commands.search import _results_to_json

    _results_to_json([])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == []

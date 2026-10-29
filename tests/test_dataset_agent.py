import csv

import pytest

from config import AppConfig, LLMConfig
from dataset_agent import (
    MAX_ATTEMPTS,
    DatasetGenerationAgent,
    choose_validation_indices,
    parse_json_response,
    save_records_as_csv,
)


def test_choose_validation_indices_checks_all_for_full_validation():
    assert choose_validation_indices(4, "full", 20) == [0, 1, 2, 3]


def test_choose_validation_indices_uses_at_least_one_random_record(monkeypatch):
    monkeypatch.setattr("dataset_agent.random.sample", lambda population, count: population[:count])

    assert choose_validation_indices(4, "random_sample", 1) == [0]


def test_parse_json_response_accepts_markdown_fence():
    assert parse_json_response('```json\n[{"query": "hi"}]\n```') == [
        {"query": "hi"}
    ]


def test_save_records_as_csv_serializes_nested_values(tmp_path):
    output_path = tmp_path / "dataset.csv"

    saved_path = save_records_as_csv(
        [{"query": "hello", "labels": ["greeting"]}],
        str(output_path),
    )

    with output_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert saved_path == str(output_path.resolve())
    assert rows == [{"query": "hello", "labels": '["greeting"]'}]


def test_save_records_as_csv_does_not_overwrite_existing_file(tmp_path):
    output_path = tmp_path / "dataset.csv"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_records_as_csv([{"query": "hello"}], str(output_path))

    assert output_path.read_text(encoding="utf-8") == "existing"


def test_agent_retries_failed_validation_then_saves(tmp_path, monkeypatch):
    validation_calls = 0

    def fake_complete(client, model, system, user):
        nonlocal validation_calls
        if "strict dataset validator" in system:
            validation_calls += 1
            if validation_calls == 1:
                return '{"valid": false, "issues": ["Missing annotation"]}'
            return '{"valid": true, "issues": []}'
        return '[{"query": "hello", "intent": "greeting"}]'

    monkeypatch.setattr(DatasetGenerationAgent, "_complete", staticmethod(fake_complete))
    llm = LLMConfig("http://localhost/v1", "EMPTY", "gemma4-26B")
    agent = DatasetGenerationAgent(AppConfig(llm, llm))
    output_path = tmp_path / "validated.csv"

    result = agent.run(
        generation_prompt="Generate a greeting query.",
        annotation_prompt="Add its intent.",
        record_count=1,
        validation_mode="full",
        sample_percent=20,
        output_path=str(output_path),
    )

    assert result["attempt"] == 2
    assert result["saved_path"] == str(output_path.resolve())
    assert validation_calls == 2
    assert output_path.exists()


def test_agent_does_not_save_after_max_failed_attempts(tmp_path, monkeypatch):
    generation_calls = 0

    def fake_complete(client, model, system, user):
        nonlocal generation_calls
        if "strict dataset validator" in system:
            return '{"valid": false, "issues": ["Does not follow instructions"]}'
        generation_calls += 1
        return '[{"query": "hello"}]'

    monkeypatch.setattr(DatasetGenerationAgent, "_complete", staticmethod(fake_complete))
    llm = LLMConfig("http://localhost/v1", "EMPTY", "gemma4-26B")
    agent = DatasetGenerationAgent(AppConfig(llm, llm))
    output_path = tmp_path / "never-created.csv"

    result = agent.run(
        generation_prompt="Generate a greeting query.",
        annotation_prompt="Add its intent.",
        record_count=1,
        validation_mode="full",
        sample_percent=20,
        output_path=str(output_path),
    )

    assert result["attempt"] == MAX_ATTEMPTS
    assert generation_calls == MAX_ATTEMPTS
    assert "saved_path" not in result
    assert not output_path.exists()

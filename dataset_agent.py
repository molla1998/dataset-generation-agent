import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from config import AppConfig, LLMConfig

MAX_ATTEMPTS = 3


class WorkflowState(TypedDict, total=False):
    generation_prompt: str
    annotation_prompt: str
    record_count: int
    validation_mode: Literal["full", "random_sample"]
    sample_percent: int
    output_path: str
    records: list[dict[str, Any]]
    attempt: int
    validation_passed: bool
    validation_results: list[dict[str, Any]]
    saved_path: str


def parse_json_response(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("The model response contained an incomplete JSON fence.")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("The model response did not contain valid JSON.") from error


def choose_validation_indices(
    record_count: int, validation_mode: str, sample_percent: int
) -> list[int]:
    validate_validation_options(record_count, validation_mode, sample_percent)
    if validation_mode == "full":
        return list(range(record_count))

    sample_count = max(1, math.ceil(record_count * sample_percent / 100))
    return sorted(random.sample(range(record_count), sample_count))


def validate_validation_options(
    record_count: int, validation_mode: str, sample_percent: int
) -> None:
    if record_count < 1:
        raise ValueError("At least one generated record is required.")
    if validation_mode not in ("full", "random_sample"):
        raise ValueError(f"Unsupported validation mode: {validation_mode}")
    if validation_mode == "random_sample" and not 1 <= sample_percent <= 100:
        raise ValueError("Random-sample validation must be between 1 and 100 percent.")


def save_records_as_csv(records: list[dict[str, Any]], output_path: str) -> str:
    if not records:
        raise ValueError("Cannot save an empty dataset.")

    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".csv":
        raise ValueError("The output filename must have a .csv extension.")
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    if not fieldnames:
        raise ValueError("Generated records do not contain any fields.")

    with path.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for record in records:
            row = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in record.items()
            }
            writer.writerow(row)
    return str(path.resolve())


class DatasetGenerationAgent:
    def __init__(self, config: AppConfig):
        self.generation_client = self._make_client(config.generation_llm)
        self.validation_client = self._make_client(config.validation_llm)
        self._generation_model = config.generation_llm.model
        self._validation_model = config.validation_llm.model
        graph = StateGraph(WorkflowState)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)
        graph.add_node("save", self._save)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", "validate")
        graph.add_conditional_edges(
            "validate",
            self._next_step,
            {"retry": "generate", "save": "save", "failed": END},
        )
        graph.add_edge("save", END)
        self.graph = graph.compile()

    @staticmethod
    def _make_client(config: LLMConfig) -> OpenAI:
        return OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=180,
        )

    @staticmethod
    def _complete(client: OpenAI, model: str, system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        if not response.choices or not isinstance(
            response.choices[0].message.content, str
        ):
            raise ValueError("The LLM returned an empty response.")
        return response.choices[0].message.content

    def _generate(self, state: WorkflowState) -> dict[str, Any]:
        attempt = state.get("attempt", 0) + 1
        instructions = (
            "Generation instructions:\n"
            f"{state['generation_prompt']}\n\n"
            "Annotation instructions:\n"
            f"{state['annotation_prompt']}"
        )
        feedback = ""
        if state.get("validation_results"):
            issues = [
                {
                    "record_index": result["index"],
                    "issues": result["issues"],
                }
                for result in state["validation_results"]
                if not result["valid"]
            ]
            feedback = (
                "\n\nThe previous batch failed validation. Correct these issues "
                "in the new batch:\n"
                f"{json.dumps(issues, ensure_ascii=False)}"
            )
        system = (
            "Generate a query-understanding dataset that follows every supplied "
            "generation and annotation instruction. Return only a JSON array of "
            "objects, with exactly the requested number of records. Do not wrap "
            "the JSON in markdown."
        )
        user = (
            f"{instructions}{feedback}\n\n"
            f"Generate exactly {state['record_count']} records."
        )
        content = self._complete(
            self.generation_client,
            self._generation_model,
            system,
            user,
        )
        parsed = parse_json_response(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            parsed = parsed["records"]
        if not isinstance(parsed, list) or not all(
            isinstance(record, dict) for record in parsed
        ):
            raise ValueError("Generation must return a JSON array of record objects.")
        if len(parsed) != state["record_count"]:
            raise ValueError(
                f"Expected {state['record_count']} records, but the model returned "
                f"{len(parsed)}."
            )
        return {"records": parsed, "attempt": attempt}

    def _validate(self, state: WorkflowState) -> dict[str, Any]:
        records = state["records"]
        indices = choose_validation_indices(
            len(records),
            state["validation_mode"],
            state.get("sample_percent", 100),
        )
        instructions = (
            "Generation instructions:\n"
            f"{state['generation_prompt']}\n\n"
            "Annotation instructions:\n"
            f"{state['annotation_prompt']}"
        )
        results = []
        for index in indices:
            system = (
                "You are a strict dataset validator. Check whether the provided "
                "record follows all generation and annotation instructions. Do not "
                "rewrite it. Return only JSON in this exact shape: "
                '{"valid": true, "issues": []}. Set valid to false and explain '
                "every unmet instruction in issues when it fails."
            )
            user = (
                f"{instructions}\n\n"
                "Record to validate:\n"
                f"{json.dumps(records[index], ensure_ascii=False)}"
            )
            parsed = parse_json_response(
                self._complete(
                    self.validation_client,
                    self._validation_model,
                    system,
                    user,
                )
            )
            if not isinstance(parsed, dict) or not isinstance(
                parsed.get("valid"), bool
            ):
                raise ValueError(
                    "The validation model response must include a boolean 'valid' field."
                )
            issues = parsed.get("issues", [])
            if not isinstance(issues, list) or not all(
                isinstance(issue, str) for issue in issues
            ):
                raise ValueError(
                    "The validation model response must include an 'issues' list of strings."
                )
            results.append(
                {"index": index, "valid": parsed["valid"], "issues": issues}
            )

        return {
            "validation_results": results,
            "validation_passed": all(result["valid"] for result in results),
        }

    @staticmethod
    def _next_step(state: WorkflowState) -> str:
        if state["validation_passed"]:
            return "save"
        if state["attempt"] < MAX_ATTEMPTS:
            return "retry"
        return "failed"

    @staticmethod
    def _save(state: WorkflowState) -> dict[str, str]:
        return {"saved_path": save_records_as_csv(state["records"], state["output_path"])}

    def run(
        self,
        generation_prompt: str,
        annotation_prompt: str,
        record_count: int,
        validation_mode: Literal["full", "random_sample"],
        sample_percent: int,
        output_path: str,
    ) -> dict[str, Any]:
        if not generation_prompt.strip():
            raise ValueError("Generation prompt is required.")
        if not annotation_prompt.strip():
            raise ValueError("Annotation prompt is required.")
        if record_count < 1:
            raise ValueError("Record count must be at least one.")
        output = Path(output_path).expanduser()
        if output.suffix.lower() != ".csv":
            raise ValueError("The output filename must have a .csv extension.")
        if output.exists():
            raise FileExistsError(f"Output file already exists: {output}")
        # Validate all user-controlled workflow options before making API calls.
        validate_validation_options(record_count, validation_mode, sample_percent)
        return self.graph.invoke(
            {
                "generation_prompt": generation_prompt,
                "annotation_prompt": annotation_prompt,
                "record_count": record_count,
                "validation_mode": validation_mode,
                "sample_percent": sample_percent,
                "output_path": output_path,
                "attempt": 0,
            }
        )

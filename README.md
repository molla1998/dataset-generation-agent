# Query Dataset Generator

A Streamlit UI backed by a LangGraph workflow for generating and annotating
query-understanding datasets, validating them with an OpenAI-compatible vLLM
endpoint, and saving passed datasets as CSV. Validation can check every record
or a random sample. The workflow retries the complete batch up to three
generation attempts and does not save a batch that fails validation.

## Setup

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `VLLM_BASE_URL`, `VLLM_API_KEY`, and `VLLM_MODEL` in `.env` for your
OpenAI-compatible vLLM server. The defaults point to `http://localhost:8000/v1`
and use `gemma4-26B`. If your server uses a different model identifier, set
`VLLM_MODEL` to the identifier it serves.

Validation uses the same endpoint, key, and model by default. To configure a
separate validation server later, set `VALIDATION_VLLM_BASE_URL`,
`VALIDATION_VLLM_API_KEY`, and/or `VALIDATION_VLLM_MODEL`.

## Run

```powershell
streamlit run app.py
```

Enter generation and annotation instructions, choose a record count and
validation mode, then provide an output directory and `.csv` filename. The
directory and file path are on the machine running Streamlit; a download button
is also provided after a successful save. Existing files are not overwritten.
Nested annotation values are serialized as JSON strings in their CSV cells.

For random-sample validation, the UI percentage is rounded up to a record count
and at least one record is checked. A validation failure causes a new complete
batch to be generated, with the validation feedback included in the next
generation prompt. After three failed attempts, no CSV is created.

## Tests

```powershell
python -m pytest
```

# Start the Query Dataset Generator

## 1. Start your vLLM server

Start the OpenAI-compatible vLLM server that serves Gemma4-26B. By default, the
app expects it at `http://localhost:8000/v1`. Keep the server running while you
use the app.

If your server uses a different address or model identifier, configure those in
the next step.

## 2. Install the app

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If `.env` already exists, keep it and edit it instead of copying over it.

## 3. Configure the vLLM connection

Open `.env` in a text editor. Set the values to match your vLLM server:

```dotenv
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=gemma4-26B
```

Use the model identifier reported by your vLLM server for `VLLM_MODEL`. If the
server requires an API key, set `VLLM_API_KEY` accordingly. Generation and
validation use the same connection by default.

## 4. Launch the Streamlit app

In the same PowerShell window, from the project folder, run:

```powershell
python -m streamlit run app.py
```

Open the **Local URL** printed in the terminal, usually
`http://localhost:8501`.

## 5. Generate a dataset

1. Enter the generation prompt and annotation prompt.
2. Choose the number of records.
3. Choose full validation or random-sample validation. For random sampling,
   choose the percentage to check.
4. Choose an output directory and a filename ending in `.csv`.
5. Select **Generate, validate, and save**.

The CSV is saved on the machine running Streamlit and can also be downloaded
from the page. A failed batch is retried up to three times; if it still fails
validation, no CSV is saved. Existing output files are not overwritten.

## Stop the app

Return to the PowerShell window running Streamlit and press `Ctrl+C`.

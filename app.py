from pathlib import Path

import streamlit as st
from openai import OpenAIError

from config import load_config
from dataset_agent import DatasetGenerationAgent, MAX_ATTEMPTS

st.set_page_config(page_title="Query Dataset Generator", page_icon="🧪")
st.title("Query Dataset Generator")
st.caption(
    "Generate and annotate query-understanding data, validate it, and save "
    "successful output as CSV."
)

config = load_config()
agent = DatasetGenerationAgent(config)

with st.form("dataset_generation_form"):
    generation_prompt = st.text_area(
        "Generation prompt",
        placeholder="Describe the query-understanding examples to generate.",
        height=160,
    )
    annotation_prompt = st.text_area(
        "Annotation prompt",
        placeholder="Describe the labels or annotations to add to each example.",
        height=160,
    )
    record_count = st.number_input(
        "Number of records",
        min_value=1,
        max_value=1000,
        value=20,
        step=1,
    )
    validation_choice = st.radio(
        "Validation",
        ("Full validation", "Random-sample validation"),
        help="Full validation checks every record. Random-sample validation checks "
        "a randomly selected percentage of records.",
    )
    sample_percent = 20
    if validation_choice == "Random-sample validation":
        sample_percent = st.slider(
            "Random sample size (%)",
            min_value=1,
            max_value=100,
            value=20,
        )
    output_directory = st.text_input(
        "Output directory (on the Streamlit host)",
        value="outputs",
    )
    output_filename = st.text_input(
        "Output filename",
        value="generated_dataset.csv",
    )
    submitted = st.form_submit_button("Generate, validate, and save")

if submitted:
    output_path = str(Path(output_directory).expanduser() / output_filename)
    validation_mode = (
        "full" if validation_choice == "Full validation" else "random_sample"
    )
    try:
        with st.spinner(
            f"Generating and validating (up to {MAX_ATTEMPTS} attempts)..."
        ):
            result = agent.run(
                generation_prompt=generation_prompt,
                annotation_prompt=annotation_prompt,
                record_count=int(record_count),
                validation_mode=validation_mode,
                sample_percent=sample_percent,
                output_path=output_path,
            )
    except (OpenAIError, ValueError, FileExistsError, OSError) as error:
        st.error(str(error))
    else:
        if result.get("saved_path"):
            st.success(f"Validated dataset saved to: {result['saved_path']}")
            with open(result["saved_path"], "rb") as dataset_file:
                st.download_button(
                    "Download generated CSV",
                    data=dataset_file.read(),
                    file_name=Path(result["saved_path"]).name,
                    mime="text/csv",
                )
        else:
            st.error(
                f"Validation did not pass after {result.get('attempt', MAX_ATTEMPTS)} "
                "generation attempts. No output file was created."
            )
            st.write("Final validation results")
            st.json(result.get("validation_results", []))

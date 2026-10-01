"""
File Processing Tools for the GAIA Agent.
Handles downloading files from the API, executing Python code, and parsing Excel files.
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd
import requests
from langchain_core.tools import tool

# Base API URL for fetching files
API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"


def download_file(task_id: str, file_name: str = "", save_dir: str = None) -> str:
    """
    Download or resolve a file associated with a task_id / file_name.
    Checks local data directories, Hugging Face Hub (GAIA dataset), and the scoring API.

    Args:
        task_id: The task ID to fetch the file for.
        file_name: The optional filename from the question metadata.
        save_dir: Directory to save the file. Uses temp dir if None.

    Returns:
        The path to the downloaded or resolved file.
    """
    # 1. Check local search paths first
    if file_name:
        candidate_paths = [
            os.path.join(os.getcwd(), "data", file_name),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file_name),
            os.path.join(os.getcwd(), file_name),
        ]
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isfile(p):
                print(f"Found local file: {p}")
                return p

    # 2. Try downloading from Hugging Face Hub (gaia-benchmark/GAIA dataset)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if file_name and hf_token:
        try:
            from huggingface_hub import hf_hub_download

            print(f"Attempting HF Hub download for {file_name} from gaia-benchmark/GAIA...")
            hub_path = hf_hub_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                filename=f"2023/validation/{file_name}",
                token=hf_token,
            )
            if hub_path and os.path.exists(hub_path):
                print(f"HF Hub download successful: {hub_path}")
                return hub_path
        except Exception as e:
            print(f"HF Hub download notice: {e}")

    # 3. Fallback to scoring API endpoint
    if save_dir is None:
        save_dir = tempfile.mkdtemp()

    url = f"{API_BASE_URL}/files/{task_id}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Try to get filename from Content-Disposition header
        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip('"').strip("'")
        elif file_name:
            filename = file_name
        else:
            # Guess from content type
            content_type = response.headers.get("Content-Type", "")
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "audio/mpeg": ".mp3",
                "audio/mp3": ".mp3",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "text/x-python": ".py",
                "application/octet-stream": ".bin",
            }
            ext = ext_map.get(content_type, ".bin")
            filename = f"{task_id}{ext}"

        filepath = os.path.join(save_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)

        return filepath

    except Exception as e:
        return f"ERROR: Failed to download file for task {task_id}: {str(e)}"


@tool
def execute_python_file(file_path: str) -> str:
    """
    Execute a Python file and return its output.
    Use this tool when the question asks about the output of an attached Python file.

    Args:
        file_path: The absolute path to the Python file to execute.

    Returns:
        The stdout output of the Python file execution.
    """
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.dirname(file_path),
        )

        output = ""
        if result.stdout:
            output += result.stdout.strip()
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr.strip()}"
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        return output if output else "No output produced."

    except subprocess.TimeoutExpired:
        return "Error: Python file execution timed out (30s limit)."
    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def execute_python_code(code: str) -> str:
    """
    Execute arbitrary Python code and return its printed stdout.
    Use this tool to calculate mathematical results, analyze tabular data, solve logic puzzles,
    sort or filter lists, verify counter-examples, or do string operations.

    Args:
        code: Valid Python code string to execute. Must print() the result.

    Returns:
        The printed stdout output or error.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = ""
        if result.stdout:
            output += result.stdout.strip()
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr.strip()}"
        return output if output else "No output produced. Make sure to use print() to output results."
    except subprocess.TimeoutExpired:
        return "Error: Python code execution timed out (15s limit)."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


@tool
def read_excel_file(file_path: str) -> str:
    """
    Read an Excel file and return its contents as a formatted string.
    Use this tool when the question involves an attached Excel/spreadsheet file.

    Args:
        file_path: The absolute path to the Excel file (.xlsx).

    Returns:
        A string representation of the Excel data, including sheet names and content.
    """
    try:
        # Read all sheets
        excel_data = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

        output_parts = []
        for sheet_name, df in excel_data.items():
            output_parts.append(f"=== Sheet: {sheet_name} ===")
            output_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            output_parts.append(f"Columns: {list(df.columns)}")
            output_parts.append("")
            output_parts.append(df.to_string(index=False))
            output_parts.append("")

            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                output_parts.append("Numeric Summary:")
                for col in numeric_cols:
                    output_parts.append(
                        f"  {col}: sum={df[col].sum():.2f}, "
                        f"mean={df[col].mean():.2f}, "
                        f"min={df[col].min()}, max={df[col].max()}"
                    )
                output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Error reading Excel file: {str(e)}"

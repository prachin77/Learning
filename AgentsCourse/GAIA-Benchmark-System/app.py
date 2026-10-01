"""
GAIA Agent — Gradio App for HuggingFace Spaces Deployment.

This is the main entry point that:
1. Provides a Gradio UI with HF login
2. Fetches GAIA benchmark questions from the scoring API
3. Runs the LangGraph agent on each question
4. Submits answers and displays the score

Based on the agents-course Final Assignment Template.
"""

import os
import time

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv

# ZeroGPU compatibility for Hugging Face Spaces
try:
    import spaces

    @spaces.GPU
    def _zerogpu_handler():
        """Satisfies ZeroGPU startup check if Space runs on ZeroGPU hardware."""
        return True
except Exception:
    pass

# Load .env for local development (ignored on HF Spaces — use Secrets there)
load_dotenv()

from agent import GAIAAgent

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GAIAAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = GAIAAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # Build agent code link
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code link: {agent_code}")

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent on Each Question
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    for i, item in enumerate(questions_data):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name", "")

        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        print(f"\n[{i+1}/{len(questions_data)}] Processing task: {task_id}")

        try:
            # Call agent with full context (question, task_id, file_name)
            submitted_answer = agent(
                question=question_text,
                task_id=task_id,
                file_name=file_name,
            )
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text[:100] + "..."
                    if len(question_text) > 100
                    else question_text,
                    "File": file_name or "—",
                    "Submitted Answer": submitted_answer,
                }
            )
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text[:100] + "...",
                    "File": file_name or "—",
                    "Submitted Answer": f"AGENT ERROR: {e}",
                }
            )

        # 4-second pause to prevent bursting the 15 RPM free tier limit
        time.sleep(4)

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(
            results_log
        )

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = (
        f"Agent finished. Submitting {len(answers_payload)} answers "
        f"for user '{username}'..."
    )
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=120)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/"
            f"{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += (
                f" Detail: {error_json.get('detail', e.response.text)}"
            )
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = (
            f"An unexpected error occurred during submission: {e}"
        )
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface ---
with gr.Blocks(
    title="GAIA Agent — Final Assignment",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🤖 GAIA Benchmark System — Final Assignment
        ### HuggingFace Agents Course — LangGraph + Groq Qwen 3.8-27B + Gemini 2.5 Flash

        ---

        **How it works:**
        1. Log in to your Hugging Face account below.
        2. Click **"Run Evaluation & Submit All Answers"** to:
           - Fetch 20 GAIA benchmark questions
           - Run the LangGraph agent (with web search, file processing, audio/image analysis)
           - Submit answers for scoring (EXACT MATCH)
           - Display your score

        **Agent Architecture:** LangGraph ReAct Agent → Groq Qwen 3.8-27B (reasoning) + Gemini 2.5 Flash (vision/fallback) + Whisper v3 (audio) → 7 Tools
        (web search, Wikipedia, page scraper, Python executor, Excel reader, YouTube transcripts, multimodal vision/audio)

        ---
        """
    )

    gr.LoginButton()

    run_button = gr.Button(
        "🚀 Run Evaluation & Submit All Answers",
        variant="primary",
        size="lg",
    )

    status_output = gr.Textbox(
        label="Run Status / Submission Result",
        lines=5,
        interactive=False,
    )
    results_table = gr.DataFrame(
        label="Questions and Agent Answers",
        wrap=True,
    )

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )

if __name__ == "__main__":
    print("\n" + "-" * 30 + " GAIA Agent Starting " + "-" * 30)

    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"   Runtime URL: https://{space_host}.hf.space")
    else:
        print("ℹ️  Running locally (SPACE_HOST not set)")

    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(
            f"   Repo: https://huggingface.co/spaces/{space_id}/tree/main"
        )
    else:
        print("ℹ️  SPACE_ID not set (running locally)")

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ Gemini API key found")
    else:
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found!")
        print("   Get a free key: https://aistudio.google.com/")

    print("-" * (60 + len(" GAIA Agent Starting ")) + "\n")
    print("Launching Gradio Interface...")
    demo.launch(ssr_mode=False)

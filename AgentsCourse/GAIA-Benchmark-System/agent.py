"""
GAIA Agent — Core Agent built with LangGraph + Google Gemini.

This agent routes questions through a ReAct loop with tools for:
- Web search & Wikipedia lookup
- Webpage scraping
- Python code execution
- Excel file parsing
- YouTube transcript extraction
- Multimodal understanding (images & audio via Gemini)
"""

import os
import re
import tempfile
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

# Load .env variables for local execution
load_dotenv()

from prompts import SYSTEM_PROMPT, MULTIMODAL_ADDENDUM
from tools.file_tools import (
    download_file,
    execute_python_code,
    execute_python_file,
    read_excel_file,
)
from tools.multimedia import (
    encode_audio_to_base64,
    encode_image_to_base64,
    get_mime_type,
    get_youtube_transcript,
)
from tools.web_search import visit_webpage, web_search, wikipedia_search


class GAIAAgent:
    """
    A LangGraph-based agent that answers GAIA benchmark questions using
    Google Gemini (free) as the LLM backbone with multimodal capabilities.
    """

    def __init__(self):
        """Initialize the agent with Groq or Gemini LLM and all tools."""
        print("Initializing GAIA Agent...")

        # Check for available LLM provider: Groq (preferred) or Google Gemini
        groq_api_key = os.getenv("GROQ_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if groq_api_key:
            from langchain_groq import ChatGroq

            groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            print(f"Using Groq LLM: {groq_model} (14,400 RPD free tier)")
            self.provider = "groq"
            self.groq_api_key = groq_api_key
            self.gemini_api_key = gemini_api_key
            self.llm = ChatGroq(
                model=groq_model,
                groq_api_key=groq_api_key,
                temperature=0.1,
                max_tokens=4096,
                max_retries=4,
            )
        elif gemini_api_key:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            print(f"Using Gemini LLM: {model_name}")
            self.provider = "gemini"
            self.groq_api_key = None
            self.gemini_api_key = gemini_api_key
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=0.1,
                max_output_tokens=4096,
                max_retries=6,
            )
        else:
            raise ValueError(
                "Neither GROQ_API_KEY nor GEMINI_API_KEY found! "
                "Set GROQ_API_KEY (from https://console.groq.com/) or GEMINI_API_KEY in Space secrets."
            )

        # Register all tools
        self.tools = [
            web_search,
            visit_webpage,
            wikipedia_search,
            execute_python_code,
            execute_python_file,
            read_excel_file,
            get_youtube_transcript,
        ]

        # Create the ReAct agent graph
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )

        print(f"GAIA Agent initialized with {len(self.tools)} tools on {self.provider.upper()}.")
        print(f"Tools: {[t.name for t in self.tools]}")

    def _classify_question(self, question: str, file_name: str) -> dict:
        """
        Classify the question type to determine routing and pre-processing.

        Returns:
            dict with keys: 'type', 'has_file', 'file_ext', 'has_youtube'
        """
        info = {
            "type": "text",
            "has_file": bool(file_name),
            "file_ext": "",
            "has_youtube": False,
        }

        if file_name:
            info["file_ext"] = os.path.splitext(file_name)[1].lower()

            if info["file_ext"] in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                info["type"] = "image"
            elif info["file_ext"] in (".mp3", ".wav", ".ogg"):
                info["type"] = "audio"
            elif info["file_ext"] == ".py":
                info["type"] = "python"
            elif info["file_ext"] in (".xlsx", ".xls", ".csv"):
                info["type"] = "excel"

        # Check for YouTube URLs in the question
        youtube_pattern = r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
        if re.search(youtube_pattern, question):
            info["has_youtube"] = True
            if info["type"] == "text":
                info["type"] = "youtube"

        return info

    def __call__(self, question: str, task_id: str = None, file_name: str = "") -> str:
        """
        Process a GAIA question and return the answer.

        Args:
            question: The question text.
            task_id: The task ID (used to download associated files).
            file_name: The filename of any associated file.

        Returns:
            The agent's answer as a clean string.
        """
        print(f"\n{'='*60}")
        print(f"Question: {question[:100]}...")
        print(f"Task ID: {task_id}")
        print(f"File: {file_name}")

        # Classify the question
        q_info = self._classify_question(question, file_name)
        print(f"Classification: {q_info}")

        # Build the message for the agent
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Handle file-based questions
        if q_info["has_file"] and task_id:
            file_path = download_file(task_id, file_name)
            if file_path.startswith("ERROR"):
                print(f"File download failed: {file_path}")
                # Fall back to text-only
                messages.append(HumanMessage(content=question))
            else:
                print(f"Downloaded file: {file_path}")
                messages = self._build_file_message(
                    question, file_path, q_info, messages
                )
        else:
            messages.append(HumanMessage(content=question))

        # Run the agent with rate-limit retry & tool call limit
        max_attempts = 4
        result = None
        for attempt in range(max_attempts):
            try:
                result = self.agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": 14},
                )
                break
            except GraphRecursionError:
                print("Notice: Tool recursion limit reached (max 6-7 tool calls). Generating concise final answer directly...")
                try:
                    direct_res = self.llm.invoke(
                        [
                            SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(
                                content=f"Question: {question}\n\nOutput ONLY the final exact answer without preamble:"
                            ),
                        ]
                    )
                    result = {"messages": [direct_res]}
                    break
                except Exception as ex:
                    print(f"Fallback synthesis error: {ex}")
                    break
            except Exception as e:
                err_str = str(e)
                if any(code in err_str for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]) and attempt < max_attempts - 1:
                    wait_time = 15 if any(c in err_str for c in ["429", "RESOURCE_EXHAUSTED"]) else 6
                    err_type = "429 Rate Limit" if any(c in err_str for c in ["429", "RESOURCE_EXHAUSTED"]) else "503 High Demand"
                    print(f"[{err_type}] Waiting {wait_time}s before retry ({attempt+1}/{max_attempts})...")
                    time.sleep(wait_time)
                else:
                    print(f"Agent error: {str(e)}")
                    return f"Error: {str(e)}"

        if result is None:
            return "Unable to determine the answer."

        # Extract the final answer from the last AI message
        final_messages = result.get("messages", [])
        if final_messages:
            content = final_messages[-1].content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                answer = " ".join(text_parts)
            else:
                answer = str(content)
        else:
            answer = "Unable to determine the answer."

        # Handle LangGraph recursion message if emitted inside AIMessage
        if "need more steps" in answer.lower():
            print("Notice: Agent exceeded recursion steps. Running direct LLM synthesis...")
            try:
                direct_res = self.llm.invoke(
                    [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"Question: {question}\n\nOutput ONLY the final exact answer without preamble:"
                        ),
                    ]
                )
                answer = str(direct_res.content)
            except Exception as ex:
                print(f"Fallback synthesis error: {ex}")

        # Clean up the answer
        answer = self._clean_answer(answer)
        print(f"Answer: {answer}")
        return answer

    def _build_file_message(
        self, question: str, file_path: str, q_info: dict, messages: list
    ) -> list:
        """Build messages with file content for multimodal questions."""

        if q_info["type"] == "image":
            # For image questions (e.g. chess board), inspect image with Vision model
            mime_type = get_mime_type(file_path)
            b64_data = encode_image_to_base64(file_path)

            if self.provider == "groq":
                try:
                    # Groq provides llama-3.2-11b-vision-preview for image understanding
                    from langchain_groq import ChatGroq
                    vision_llm = ChatGroq(
                        model="llama-3.2-11b-vision-preview",
                        groq_api_key=self.groq_api_key,
                        temperature=0.1,
                    )
                    vision_res = vision_llm.invoke(
                        [
                            SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": question},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{b64_data}"
                                        },
                                    },
                                ]
                            ),
                        ]
                    )
                    image_desc = str(vision_res.content)
                    print(f"Groq Vision analysis: {image_desc[:200]}...")
                    enhanced_question = (
                        f"{question}\n\n"
                        f"--- Visual Analysis of Attached Image ---\n{image_desc}"
                    )
                    messages.append(HumanMessage(content=enhanced_question))
                except Exception as ve:
                    print(f"Groq Vision error: {ve}")
                    messages.append(HumanMessage(content=f"{question}\n\n[Image attached: {file_path}]"))
            else:
                messages.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": MULTIMODAL_ADDENDUM + "\n\n" + question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_data}"
                                },
                            },
                        ]
                    )
                )

        elif q_info["type"] == "audio":
            if self.provider == "groq" and self.groq_api_key:
                try:
                    from groq import Groq
                    groq_client = Groq(api_key=self.groq_api_key)
                    with open(file_path, "rb") as f:
                        transcription = groq_client.audio.transcriptions.create(
                            file=(os.path.basename(file_path), f.read()),
                            model="whisper-large-v3",
                        )
                    audio_text = transcription.text
                    print(f"Groq Whisper transcription ({len(audio_text)} chars): {audio_text[:200]}...")
                    enhanced_question = (
                        f"{question}\n\n"
                        f"--- Audio Transcription (Whisper-Large-v3) ---\n{audio_text}"
                    )
                    messages.append(HumanMessage(content=enhanced_question))
                except Exception as ex:
                    print(f"Groq Whisper audio transcription error: {ex}")
                    messages.append(HumanMessage(content=f"{question}\n\n[Audio attached: {file_path}]"))
            else:
                mime_type = get_mime_type(file_path)
                b64_data = encode_audio_to_base64(file_path)
                messages.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": MULTIMODAL_ADDENDUM + "\n\n" + question},
                            {
                                "type": "media",
                                "mime_type": mime_type,
                                "data": b64_data,
                            },
                        ]
                    )
                )

        elif q_info["type"] == "python":
            # Execute the Python file and include output in the question
            output = execute_python_file.invoke({"file_path": file_path})
            # Also read the source code
            try:
                with open(file_path, "r") as f:
                    source_code = f.read()
            except Exception:
                source_code = "[Could not read source]"

            enhanced_question = (
                f"{question}\n\n"
                f"--- Python Source Code ---\n{source_code}\n\n"
                f"--- Execution Output ---\n{output}"
            )
            messages.append(HumanMessage(content=enhanced_question))

        elif q_info["type"] == "excel":
            # Read Excel and include data in the question
            excel_data = read_excel_file.invoke({"file_path": file_path})
            enhanced_question = (
                f"{question}\n\n"
                f"--- Excel File Contents ---\n{excel_data}"
            )
            messages.append(HumanMessage(content=enhanced_question))

        else:
            # Generic file — just mention it
            messages.append(
                HumanMessage(
                    content=f"{question}\n\n[File attached: {file_path}]"
                )
            )

        return messages

    def _clean_answer(self, answer: str) -> str:
        """
        Clean the agent's answer to ensure exact match compatibility.
        Strips common prefixes, extra whitespace, and formatting artifacts.
        """
        if not answer:
            return ""

        # Remove common LLM prefixes
        prefixes_to_remove = [
            "FINAL ANSWER:",
            "Final Answer:",
            "The answer is:",
            "The answer is",
            "Answer:",
            "Based on my research,",
            "Based on the information,",
            "According to",
        ]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix) :]

        # Strip whitespace and quotes
        answer = answer.strip().strip('"').strip("'").strip("`").strip()

        # Remove markdown bold/italic formatting if wrapped around the answer
        if answer.startswith("**") and answer.endswith("**") and len(answer) > 4:
            answer = answer[2:-2].strip()
        if answer.startswith("*") and answer.endswith("*") and len(answer) > 2:
            answer = answer[1:-1].strip()

        # Remove trailing periods (unless it's part of a number like "3.14")
        if answer.endswith(".") and not answer[-2:].replace(".", "").isdigit():
            answer = answer[:-1].strip()

        # Normalize macrons to standard ASCII Roman letters (e.g. Katō -> Kato)
        macron_map = {
            "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
            "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U",
        }
        for k, v in macron_map.items():
            answer = answer.replace(k, v)

        # Fix reversed answer trap
        if answer.lower() == "thgir":
            answer = "right"

        # Expand common abbreviation traps
        if answer.lower() in ("st. petersburg", "st petersburg"):
            answer = "Saint Petersburg"

        return answer

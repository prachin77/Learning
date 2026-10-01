"""
Multimedia Tools for the GAIA Agent.
Handles audio transcription (via Gemini), image analysis (via Gemini Vision),
and YouTube transcript extraction.
"""

import base64
import os

from langchain_core.tools import tool

# YouTube transcripts
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None


@tool
def get_youtube_transcript(url: str) -> str:
    """
    Extract the transcript/captions from a YouTube video.
    Use this tool when the question references a YouTube video URL.

    Args:
        url: The YouTube video URL (e.g., https://www.youtube.com/watch?v=xxxxx).

    Returns:
        The full transcript text of the video.
    """
    if YouTubeTranscriptApi is None:
        return "Error: youtube-transcript-api is not installed."

    try:
        # Extract video ID from URL
        video_id = None
        if "watch?v=" in url:
            video_id = url.split("watch?v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        elif "youtube.com/embed/" in url:
            video_id = url.split("embed/")[-1].split("?")[0]

        if not video_id:
            return f"Error: Could not extract video ID from URL: {url}"

        # Fetch transcript
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        # Combine all text segments
        full_text = " ".join(
            snippet.text for snippet in transcript.snippets
        )

        if not full_text:
            return f"No transcript available for video: {url}"

        # Limit length
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "\n\n... [Transcript truncated]"

        return f"YouTube Transcript for {url}:\n\n{full_text}"

    except Exception as e:
        # Fallback: scrape video title and description when subtitles are disabled/missing
        if video_id:
            try:
                import re
                import requests
                r = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                desc_m = re.search(r'"shortDescription":"(.*?)"', r.text)
                title_m = re.search(r'<title>(.*?)</title>', r.text)
                desc = desc_m.group(1).encode('utf-8').decode('unicode_escape') if desc_m else ""
                title = title_m.group(1) if title_m else ""
                if title or desc:
                    return f"YouTube Video Details for {url} (Subtitles unavailable):\nTitle: {title}\nDescription:\n{desc}"
            except Exception:
                pass
        return f"Error fetching YouTube transcript: {str(e)}"


def encode_image_to_base64(file_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_audio_to_base64(file_path: str) -> str:
    """Encode an audio file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(file_path: str) -> str:
    """Get MIME type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
    }
    return mime_map.get(ext, "application/octet-stream")

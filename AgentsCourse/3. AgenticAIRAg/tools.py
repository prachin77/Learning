"""Additional tools used by the gala assistant."""

import random

from huggingface_hub import list_models
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool


search_tool = DuckDuckGoSearchRun()


def get_weather_info(location: str) -> str:
    """Return sample weather information for a location."""
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
    ]
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches sample weather information for a given location.",
)


def get_hub_stats(author: str) -> str:
    """Return the most downloaded Hugging Face model by an author."""
    try:
        models = list(
            list_models(author=author, sort="downloads", limit=1)
        )
        if not models:
            return f"No models found for author {author}."

        model = models[0]
        downloads = model.downloads or 0
        return (
            f"The most downloaded model by {author} is {model.id} "
            f"with {downloads:,} downloads."
        )
    except Exception as error:
        return f"Error fetching models for {author}: {error}"


hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description=(
        "Fetches the most downloaded model from a specific Hugging Face Hub author."
    ),
)
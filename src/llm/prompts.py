"""Prompt templates and curated few-shot examples for sentiment classification."""
import json
from typing import Any, Dict, List

# System role and class definitions
SYSTEM_INSTRUCTION = """You are an expert sentiment analysis system.
Your task is to classify customer reviews into one of three sentiment classes:
- Negative (label 0): Expresses dissatisfaction, complaints, poor service, bad food, regret, or negative experience.
- Neutral (label 1): Expresses mixed sentiments (both positive and negative points), mediocre/average satisfaction, or factual description without strong emotion.
- Positive (label 2): Expresses satisfaction, praise, delight, high recommendation, or positive experience.

You must ALWAYS respond with a strictly valid JSON object matching this exact schema:
{
  "sentiment": "Negative" | "Neutral" | "Positive",
  "label": 0 | 1 | 2,
  "confidence": <float between 0.0 and 1.0 reflecting your self-reported certainty>,
  "reason": "<one concise sentence explaining the classification>"
}

Do not wrap your output in markdown code blocks like ```json ... ```. Output raw JSON only."""

# Curated few-shot examples (strictly sequestered; not taken from the evaluation set)
FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "review": "The pasta was completely cold and the waiter ignored us for 40 minutes. Absolutely awful.",
        "output": {
            "sentiment": "Negative",
            "label": 0,
            "confidence": 0.98,
            "reason": "Expresses strong dissatisfaction regarding cold food and severe lack of service."
        }
    },
    {
        "review": "Overpriced, tiny portions, and the bathroom was filthy. Will never return.",
        "output": {
            "sentiment": "Negative",
            "label": 0,
            "confidence": 0.95,
            "reason": "Clear negative sentiment regarding poor value, cleanliness issues, and explicit intent not to return."
        }
    },
    {
        "review": "The burger was decent and fries were hot, but the drink was flat. Standard diner food, neither great nor terrible.",
        "output": {
            "sentiment": "Neutral",
            "label": 1,
            "confidence": 0.88,
            "reason": "Contains balanced positive and negative aspects resulting in an average, mediocre evaluation."
        }
    },
    {
        "review": "It was an average dining experience. The prices are standard for downtown and seating was adequate.",
        "output": {
            "sentiment": "Neutral",
            "label": 1,
            "confidence": 0.85,
            "reason": "Objective, factual description of an unremarkable experience with no strong emotional polarity."
        }
    },
    {
        "review": "Outstanding dining experience! The wood-fired pizza was crispy and delicious, and the staff made us feel like family.",
        "output": {
            "sentiment": "Positive",
            "label": 2,
            "confidence": 0.98,
            "reason": "Enthusiastic praise for food quality, hospitality, and overall experience."
        }
    },
    {
        "review": "Best coffee in town! Fast Wi-Fi, friendly baristas, and lovely ambient music. Highly recommended!",
        "output": {
            "sentiment": "Positive",
            "label": 2,
            "confidence": 0.96,
            "reason": "High praise across service, product, atmosphere, and an explicit recommendation."
        }
    }
]


def format_zero_shot_prompt(review: str) -> str:
    """Format zero-shot sentiment classification prompt."""
    return f"""Analyze the sentiment of the following customer review and output the structured JSON response:

Review: "{review}"

JSON Response:"""


def format_few_shot_prompt(review: str) -> str:
    """Format few-shot sentiment classification prompt with representative examples."""
    prompt_parts = ["Here are several examples of correctly classified customer reviews:\n"]
    
    for ex in FEW_SHOT_EXAMPLES:
        prompt_parts.append(f'Review: "{ex["review"]}"')
        prompt_parts.append(f'JSON Response: {json.dumps(ex["output"])}\n')
        
    prompt_parts.append("Now classify this new customer review:")
    prompt_parts.append(f'Review: "{review}"')
    prompt_parts.append("JSON Response:")
    
    return "\n".join(prompt_parts)


# Direct functional aliases for backward compatibility and uniform API access
build_zero_shot_prompt = format_zero_shot_prompt
build_few_shot_prompt = format_few_shot_prompt

__all__ = [
    "SYSTEM_INSTRUCTION",
    "FEW_SHOT_EXAMPLES",
    "format_zero_shot_prompt",
    "format_few_shot_prompt",
    "build_zero_shot_prompt",
    "build_few_shot_prompt",
]



"""Data preprocessing, cleaning, label mapping, and standardization."""
import html
import re
from typing import Optional, Union
import pandas as pd
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)


def clean_text(text: Optional[str]) -> str:
    """Clean and normalize raw review text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Replace HTML break tags with space
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    
    # 2. Strip other HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Unescape HTML entities (&amp;, &lt;, &gt;, etc.)
    text = html.unescape(text)
    
    # 4. Normalize whitespace (spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def map_star_to_three_class(rating: Union[int, float]) -> int:
    """Map star rating or 0-indexed rating to 3 classes:
    
    0 = Negative (1-2 stars or 0-1 0-indexed)
    1 = Neutral (3 stars or 2 0-indexed)
    2 = Positive (4-5 stars or 3-4 0-indexed)
    """
    try:
        val = int(rating)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid rating value: {rating}")
        
    # Hugging Face yelp_review_full uses 0-indexed labels: 0, 1, 2, 3, 4
    if val in (0, 1):
        return 0  # Negative
    elif val == 2:
        return 1  # Neutral
    elif val in (3, 4):
        return 2  # Positive
    # If 1-indexed (1 to 5 stars)
    elif val in (1, 2):
        return 0  # Negative
    elif val == 3:
        return 1  # Neutral
    elif val in (4, 5):
        return 2  # Positive
    else:
        raise ValueError(f"Rating {rating} outside expected range [0-4] or [1-5]")


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    source_tag: str = "real"
) -> pd.DataFrame:
    """Standardize, clean, map labels, and deduplicate a dataframe.
    
    Standard output columns:
        review (str): Cleaned review text
        sentiment (str): 'Negative', 'Neutral', 'Positive'
        label (int): 0, 1, 2
        source (str): 'real' or 'synthetic'
    """
    logger.info("Preprocessing dataframe with %d initial rows...", len(df))
    
    # Verify required columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Expected columns '{text_col}' and '{label_col}', found {list(df.columns)}")
    
    # Extract working copy
    processed = pd.DataFrame()
    processed["review"] = df[text_col].apply(clean_text)
    
    # Map to 3-class label
    processed["label"] = df[label_col].apply(map_star_to_three_class)
    processed["sentiment"] = processed["label"].map(settings.label_to_sentiment)
    processed["source"] = source_tag
    
    # Filter out empty or whitespace-only reviews
    initial_count = len(processed)
    processed = processed[processed["review"].str.len() > 0]
    empty_removed = initial_count - len(processed)
    if empty_removed > 0:
        logger.info("Removed %d empty/whitespace-only reviews.", empty_removed)
        
    # Drop duplicates on the review text
    before_dedup = len(processed)
    processed = processed.drop_duplicates(subset=["review"]).reset_index(drop=True)
    duplicates_removed = before_dedup - len(processed)
    if duplicates_removed > 0:
        logger.info("Removed %d duplicate reviews.", duplicates_removed)
        
    logger.info("Preprocessing complete. Retained %d valid unique rows.", len(processed))
    return processed

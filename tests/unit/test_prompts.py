"""Regression unit tests for prompt formatting and builder interfaces (Phase 5)."""
import json
import pytest
from src.llm.prompts import (
    SYSTEM_INSTRUCTION,
    FEW_SHOT_EXAMPLES,
    format_zero_shot_prompt,
    format_few_shot_prompt,
    build_zero_shot_prompt,
    build_few_shot_prompt,
)


class TestPromptBuilders:
    """Test suite ensuring prompt builders are importable, consistent, and leak-free."""

    def test_import_both_naming_conventions(self):
        """Verify both format_* and build_* functions are importable and identical."""
        assert callable(format_zero_shot_prompt)
        assert callable(format_few_shot_prompt)
        assert callable(build_zero_shot_prompt)
        assert callable(build_few_shot_prompt)
        assert format_zero_shot_prompt is build_zero_shot_prompt
        assert format_few_shot_prompt is build_few_shot_prompt

    def test_zero_shot_prompt_output(self):
        """Verify zero-shot prompt construction includes review text and JSON prompt."""
        review = "The artisanal espresso was rich, creamy, and served with a smile."
        prompt_format = format_zero_shot_prompt(review)
        prompt_build = build_zero_shot_prompt(review)
        
        assert prompt_format == prompt_build
        assert review in prompt_build
        assert "Review:" in prompt_build
        assert "JSON Response:" in prompt_build
        assert "Analyze the sentiment" in prompt_build

    def test_few_shot_prompt_output_structure(self):
        """Verify few-shot prompt output contains examples, instructions, and target review."""
        review = "Overcooked fries and warm soda."
        prompt_format = format_few_shot_prompt(review)
        prompt_build = build_few_shot_prompt(review)
        
        assert prompt_format == prompt_build
        assert review in prompt_build
        assert "Here are several examples" in prompt_build
        assert "Now classify this new customer review:" in prompt_build

    def test_few_shot_examples_balance_and_count(self):
        """Verify exactly 6 curated examples with 2 per class (Negative=0, Neutral=1, Positive=2)."""
        assert len(FEW_SHOT_EXAMPLES) == 6
        
        labels = [ex["output"]["label"] for ex in FEW_SHOT_EXAMPLES]
        sentiments = [ex["output"]["sentiment"] for ex in FEW_SHOT_EXAMPLES]
        
        assert labels.count(0) == 2
        assert labels.count(1) == 2
        assert labels.count(2) == 2
        
        assert sentiments.count("Negative") == 2
        assert sentiments.count("Neutral") == 2
        assert sentiments.count("Positive") == 2

    def test_few_shot_examples_schema_validity(self):
        """Verify all curated few-shot examples adhere strictly to the target JSON schema."""
        for ex in FEW_SHOT_EXAMPLES:
            assert "review" in ex and len(ex["review"]) > 10
            out = ex["output"]
            assert "sentiment" in out and out["sentiment"] in ("Negative", "Neutral", "Positive")
            assert "label" in out and out["label"] in (0, 1, 2)
            assert "confidence" in out and 0.0 <= out["confidence"] <= 1.0
            assert "reason" in out and len(out["reason"]) > 5

    def test_no_benchmark_leakage(self):
        """Verify curated few-shot examples do not collide with benchmark review texts."""
        few_shot_texts = {ex["review"].strip().lower() for ex in FEW_SHOT_EXAMPLES}
        
        # Check against evaluation dataset if present
        from config.settings import settings
        eval_path = settings.data_dir / "eval_subset_real.parquet"
        if eval_path.exists():
            import pandas as pd
            eval_df = pd.read_parquet(eval_path)
            eval_texts = set(eval_df["review"].str.strip().str.lower())
            intersection = few_shot_texts.intersection(eval_texts)
            assert len(intersection) == 0, f"Leakage detected! Colliding reviews: {intersection}"

    def test_edge_cases_handling(self):
        """Verify prompt builders gracefully handle empty, multiline, or special character reviews."""
        empty_prompt = build_zero_shot_prompt("")
        assert "Review: \"\"" in empty_prompt
        
        special_prompt = build_few_shot_prompt("Quotes: \"special\" & symbols <tag>.")
        assert "Quotes: \"special\" & symbols <tag>." in special_prompt


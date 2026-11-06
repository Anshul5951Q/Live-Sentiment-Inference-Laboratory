"""Unit tests for TextTokenizer sequence conversion, padding, and OOV handling."""
import pytest
import numpy as np
from src.models.text_tokenizer import TextTokenizer


class TestTextTokenizer:
    """Test suite for TextTokenizer."""

    @pytest.fixture
    def sample_texts(self):
        return [
            "The pizza was absolutely delicious and hot!",
            "Terrible customer service and cold food.",
            "Average lunch, mediocre burgers.",
            "Fantastic ambiance and delicious pasta.",
        ]

    def test_fit_and_vocabulary(self, sample_texts):
        tokenizer = TextTokenizer(vocab_size=20, max_length=10)
        tokenizer.fit(sample_texts)
        
        assert tokenizer.is_fitted
        # Reserved: PAD=0, OOV=1
        assert tokenizer.word_index["<PAD>"] == 0
        assert tokenizer.word_index["<OOV>"] == 1
        assert len(tokenizer.word_index) <= 20
        assert "delicious" in tokenizer.word_index

    def test_oov_handling(self, sample_texts):
        tokenizer = TextTokenizer(vocab_size=10, max_length=10)
        tokenizer.fit(sample_texts)
        
        # Test novel words never seen during training
        novel_texts = ["Zanzibar xylophone completely unknown"]
        seqs = tokenizer.texts_to_sequences(novel_texts)
        
        # All four words must map to OOV index 1
        assert seqs[0] == [1, 1, 1, 1]

    def test_padding_and_truncation(self, sample_texts):
        tokenizer = TextTokenizer(vocab_size=20, max_length=6)
        tokenizer.fit(sample_texts)
        
        # 1. Padding shorter sequence
        short_seq = [[2, 3]]
        padded = tokenizer.pad_sequences(short_seq, padding="post")
        assert padded.shape == (1, 6)
        np.testing.assert_array_equal(padded[0], [2, 3, 0, 0, 0, 0])
        
        # 2. Truncating longer sequence
        long_seq = [[2, 3, 4, 5, 6, 7, 8, 9]]
        truncated = tokenizer.pad_sequences(long_seq, truncating="post")
        assert truncated.shape == (1, 6)
        np.testing.assert_array_equal(truncated[0], [2, 3, 4, 5, 6, 7])

    def test_empty_and_none_text_handling(self, sample_texts):
        tokenizer = TextTokenizer(vocab_size=20, max_length=5)
        tokenizer.fit(sample_texts)
        
        padded = tokenizer.transform(["", "   \t\n", None])
        assert padded.shape == (3, 5)
        # All elements should be 0 (<PAD>)
        assert np.all(padded == 0)

    def test_save_and_load(self, sample_texts, tmp_path):
        tokenizer = TextTokenizer(vocab_size=20, max_length=8)
        tokenizer.fit(sample_texts)
        
        save_file = tmp_path / "tokenizer.json"
        tokenizer.save(save_file)
        
        loaded = TextTokenizer().load(save_file)
        assert loaded.is_fitted
        assert loaded.word_index == tokenizer.word_index
        assert loaded.max_length == tokenizer.max_length
        
        sample = ["Delicious pasta!"]
        np.testing.assert_array_equal(
            tokenizer.transform(sample),
            loaded.transform(sample)
        )

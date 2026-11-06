"""Unit tests for TF-IDF feature extraction and leakage prevention."""
import pytest
import numpy as np
from src.models.tfidf_vectorizer import TextTfidfVectorizer


class TestTfidfVectorizer:
    """Test suite for TextTfidfVectorizer."""

    @pytest.fixture
    def train_corpus(self):
        return [
            "Outstanding dining and excellent food!",
            "Terrible service and awful atmosphere.",
            "Average lunch, mediocre burgers.",
            "Highly recommended pizza and great wine.",
            "Worst dinner ever, dirty tables.",
            "Standard experience, nothing special.",
        ]

    def test_fit_and_transform(self, train_corpus):
        vec = TextTfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
        X_train = vec.fit_transform(train_corpus)
        
        assert vec.is_fitted
        assert X_train.shape[0] == len(train_corpus)
        assert X_train.shape[1] <= 50

    def test_vocabulary_unchanged_on_transform(self, train_corpus):
        vec = TextTfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
        vec.fit(train_corpus)
        initial_vocab = dict(vec.vectorizer.vocabulary_)
        
        # Transform novel validation/evaluation text with out-of-vocabulary words
        new_texts = ["Completely new words never seen before in train!", "Another unfamiliar review."]
        X_new = vec.transform(new_texts)
        
        # Assert vocabulary is strictly identical
        assert vec.vectorizer.vocabulary_ == initial_vocab
        assert X_new.shape[1] == len(initial_vocab)

    def test_empty_and_whitespace_handling(self, train_corpus):
        vec = TextTfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
        vec.fit(train_corpus)
        
        # Test empty string, whitespace only, and None
        sparse_out = vec.transform(["", "   \t\n  ", None])
        assert sparse_out.shape[0] == 3
        # Should produce all zeros without raising an exception
        assert sparse_out.nnz == 0

    def test_save_and_load_consistency(self, train_corpus, tmp_path):
        vec = TextTfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
        vec.fit(train_corpus)
        
        save_path = tmp_path / "tfidf.joblib"
        vec.save(save_path)
        
        loaded_vec = TextTfidfVectorizer().load(save_path)
        assert loaded_vec.is_fitted
        assert loaded_vec.vectorizer.vocabulary_ == vec.vectorizer.vocabulary_
        
        # Test transform output parity
        test_text = ["Excellent pizza!"]
        np.testing.assert_array_almost_equal(
            vec.transform(test_text).toarray(),
            loaded_vec.transform(test_text).toarray()
        )


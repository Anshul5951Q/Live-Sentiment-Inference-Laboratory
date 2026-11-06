"""Pytest fixtures and test configurations for SentimentLab AI."""
import matplotlib
matplotlib.use("Agg")

import pytest
import pandas as pd


@pytest.fixture
def sample_raw_yelp_df() -> pd.DataFrame:
    """Generate a sample raw dataframe mimicking Yelp Review Full format."""
    data = []
    # Generate 30 samples: 6 for each of 0, 1, 2, 3, 4
    texts_by_rating = {
        0: [  # 1 star -> Negative
            "Terrible experience, food was cold.<br>Never coming back!",
            "Worst place in town &amp; rude staff.",
            "Absolutely horrible service and disgusting food.",
            "Dirty tables and stale ingredients. Zero stars.",
            "The waiter yelled at us. Avoid at all costs!",
            "Completely tasteless and overpriced."
        ],
        1: [  # 2 stars -> Negative
            "Mediocre food and slow service.",
            "Not good at all, wouldn't recommend.",
            "Expected much better for the price.",
            "Disappointing portion sizes and lukewarm soup.",
            "Staff was polite, but food was terrible.",
            "Bland flavors and dirty restroom."
        ],
        2: [  # 3 stars -> Neutral
            "Average place. Food was okay, nothing special.",
            "Decent service, ordinary burgers.",
            "It was fine. Neither good nor bad.",
            "Fair experience, standard diner food.",
            "Not bad, but not memorable either.",
            "Okay for a quick lunch if nothing else is open."
        ],
        3: [  # 4 stars -> Positive
            "Good food and friendly staff. Enjoyed the pasta.",
            "Solid restaurant with nice ambiance.",
            "Great appetizers and prompt service.",
            "Tasty desserts and cozy seating.",
            "Very pleasant dinner with family.",
            "Above average quality and fair prices."
        ],
        4: [  # 5 stars -> Positive
            "Outstanding dining! Best steak I ever had!",
            "Exceptional customer service and amazing pizza!",
            "Incredible culinary experience! 10/10!",
            "Perfection in every bite. Highly recommend!",
            "The chef is a master. Phenomenal evening!",
            "Delicious food, superb cocktails, wonderful vibe!"
        ],
    }
    
    for rating, texts in texts_by_rating.items():
        for t in texts:
            data.append({"text": t, "label": rating})
            
    return pd.DataFrame(data)


@pytest.fixture
def sample_duplicates_df() -> pd.DataFrame:
    """Sample DataFrame with duplicates and HTML tags."""
    return pd.DataFrame([
        {"text": "<b>Great!</b> Best food ever! &amp; drinks.", "label": 4},
        {"text": "<b>Great!</b> Best food ever! &amp; drinks.", "label": 4},  # Exact duplicate
        {"text": "   ", "label": 2},  # Empty whitespace
        {"text": "Terrible place.<br/><br/>", "label": 0},
    ])


"""
Sentiment Analysis for Trading

Extract market sentiment from:
- News headlines (Bloomberg, Reuters, CNBC)
- Social media (Twitter/X FinTwit, Reddit WSB)
- Fed speeches (Hawkish/Dovish classification)
- Analyst reports

Uses FinBERT - BERT fine-tuned on financial text.
"""

import numpy as np
import logging
from datetime import datetime
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Market sentiment analysis

    Note: This is a template. In production, you would:
    1. Install transformers: pip install transformers
    2. Use FinBERT: "ProsusAI/finbert"
    3. Scrape news from APIs (NewsAPI, Alpha Vantage, etc.)
    4. Monitor social media (Twitter API, Reddit PRAW)
    """

    def __init__(self, use_finbert=False):
        """
        Initialize sentiment analyzer

        Args:
            use_finbert: If True, load FinBERT model (requires transformers package)
        """

        self.use_finbert = use_finbert
        self.model = None
        self.tokenizer = None

        if use_finbert:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                logger.info("✅ FinBERT model loaded")
            except ImportError:
                logger.warning("⚠️ transformers not installed - using keyword-based sentiment")
                self.use_finbert = False

        # Sentiment history
        self.sentiment_history = deque(maxlen=100)

        logger.info("📰 Sentiment Analyzer initialized")

    def analyze_headlines(self, headlines):
        """
        Analyze sentiment of news headlines

        Args:
            headlines: List of headline strings

        Returns:
            sentiment_score: -1 (bearish) to +1 (bullish)
        """

        if not headlines:
            return 0.0

        if self.use_finbert and self.model is not None:
            return self._analyze_with_finbert(headlines)
        else:
            return self._analyze_with_keywords(headlines)

    def _analyze_with_finbert(self, headlines):
        """Use FinBERT for sentiment analysis"""

        import torch

        sentiments = []

        for headline in headlines:
            # Tokenize
            inputs = self.tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)

            # Get sentiment
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [negative, neutral, positive]
            negative = probs[0][0].item()
            positive = probs[0][2].item()

            # Score: -1 (bearish) to +1 (bullish)
            sentiment_score = positive - negative

            sentiments.append(sentiment_score)

        return np.mean(sentiments) if sentiments else 0.0

    def _analyze_with_keywords(self, headlines):
        """Keyword-based sentiment (fallback)"""

        bullish_keywords = [
            'surge', 'rally', 'gain', 'rise', 'jump', 'soar',
            'bullish', 'breakout', 'strength', 'support',
            'demand', 'optimistic', 'positive', 'growth'
        ]

        bearish_keywords = [
            'plunge', 'crash', 'drop', 'fall', 'decline', 'slide',
            'bearish', 'breakdown', 'weakness', 'resistance',
            'fear', 'pessimistic', 'negative', 'recession'
        ]

        bullish_score = 0
        bearish_score = 0

        for headline in headlines:
            headline_lower = headline.lower()

            # Count bullish keywords
            bullish_score += sum(1 for kw in bullish_keywords if kw in headline_lower)

            # Count bearish keywords
            bearish_score += sum(1 for kw in bearish_keywords if kw in headline_lower)

        total = bullish_score + bearish_score

        if total == 0:
            return 0.0

        # Normalize to -1 to +1
        return (bullish_score - bearish_score) / total

    def analyze_fed_speech(self, speech_text):
        """
        Analyze Fed speech for hawkish/dovish sentiment

        Hawkish = raise rates = strong USD = bearish Gold
        Dovish = easy money = weak USD = bullish Gold

        Returns:
            sentiment: -1 (hawkish) to +1 (dovish)
        """

        hawkish_keywords = [
            'inflation', 'raise rates', 'tighten', 'hawkish',
            'strength', 'resilient', 'overheating', 'persistent',
            'restrictive', 'combat inflation'
        ]

        dovish_keywords = [
            'stimulus', 'support', 'dovish', 'patient',
            'accommodative', 'weakness', 'downside risks',
            'monitor', 'gradual', 'data dependent'
        ]

        text_lower = speech_text.lower()

        hawkish_score = sum(text_lower.count(kw) for kw in hawkish_keywords)
        dovish_score = sum(text_lower.count(kw) for kw in dovish_keywords)

        total = hawkish_score + dovish_score

        if total == 0:
            return 0.0

        # Normalize: +1 = dovish (bullish Gold), -1 = hawkish (bearish Gold)
        return (dovish_score - hawkish_score) / total

    def get_social_sentiment(self, keywords=['gold', 'xauusd']):
        """
        Get sentiment from social media (Twitter, Reddit)

        Note: This is a placeholder. In production:
        1. Use Twitter API v2
        2. Use Reddit PRAW
        3. Monitor r/wallstreetbets, r/gold, FinTwit
        """

        # Placeholder - would scrape real data
        return 0.0  # Neutral

    def aggregate_sentiment(self, news_headlines=None, fed_text=None):
        """
        Aggregate sentiment from all sources

        Returns:
            features: Dict with sentiment features
        """

        features = {}

        # News sentiment
        if news_headlines:
            features['news_sentiment'] = self.analyze_headlines(news_headlines)
        else:
            features['news_sentiment'] = 0.0

        # Fed sentiment
        if fed_text:
            features['fed_sentiment'] = self.analyze_fed_speech(fed_text)
        else:
            features['fed_sentiment'] = 0.0

        # Social sentiment
        features['social_sentiment'] = self.get_social_sentiment()

        # Overall sentiment (weighted average)
        features['overall_sentiment'] = (
            0.5 * features['news_sentiment'] +
            0.3 * features['fed_sentiment'] +
            0.2 * features['social_sentiment']
        )

        # Sentiment momentum (change from previous)
        if self.sentiment_history:
            prev_sentiment = self.sentiment_history[-1]['overall_sentiment']
            features['sentiment_momentum'] = features['overall_sentiment'] - prev_sentiment
        else:
            features['sentiment_momentum'] = 0.0

        # Sentiment divergence (news vs social)
        features['sentiment_divergence'] = features['news_sentiment'] - features['social_sentiment']

        # Store history
        self.sentiment_history.append(features.copy())

        return features


# Example usage
if __name__ == "__main__":
    print("📰 Sentiment Analysis Demo\n")

    # Create analyzer
    analyzer = SentimentAnalyzer(use_finbert=False)  # Set True if transformers installed

    # Test 1: News headlines
    print("="*60)
    print("Test 1: News Headlines")
    print("="*60)

    bullish_headlines = [
        "Gold surges to record high on safe-haven demand",
        "Analysts bullish on precious metals amid uncertainty",
        "Gold rallies as dollar weakens"
    ]

    bearish_headlines = [
        "Gold plunges as dollar strengthens",
        "Analysts turn bearish on gold outlook",
        "Gold crashes on Fed hawkish stance"
    ]

    neutral_headlines = [
        "Gold trading flat in Asian session",
        "Markets await Fed decision"
    ]

    bullish_score = analyzer.analyze_headlines(bullish_headlines)
    bearish_score = analyzer.analyze_headlines(bearish_headlines)
    neutral_score = analyzer.analyze_headlines(neutral_headlines)

    print(f"Bullish headlines sentiment: {bullish_score:+.3f}")
    print(f"Bearish headlines sentiment: {bearish_score:+.3f}")
    print(f"Neutral headlines sentiment: {neutral_score:+.3f}")

    # Test 2: Fed speech
    print("\n" + "="*60)
    print("Test 2: Fed Speech Analysis")
    print("="*60)

    hawkish_speech = """
    The Federal Reserve remains committed to combating inflation.
    We will raise rates as needed to ensure price stability.
    The economy shows resilient strength despite our tightening.
    """

    dovish_speech = """
    The Federal Reserve will remain patient and data dependent.
    We see downside risks to economic growth.
    We will support the economy with accommodative policy.
    """

    hawkish_score = analyzer.analyze_fed_speech(hawkish_speech)
    dovish_score = analyzer.analyze_fed_speech(dovish_speech)

    print(f"Hawkish speech: {hawkish_score:+.3f} (negative for Gold)")
    print(f"Dovish speech: {dovish_score:+.3f} (positive for Gold)")

    # Test 3: Aggregate
    print("\n" + "="*60)
    print("Test 3: Aggregate Sentiment")
    print("="*60)

    features = analyzer.aggregate_sentiment(
        news_headlines=bullish_headlines,
        fed_text=dovish_speech
    )

    print("Sentiment features:")
    for key, value in features.items():
        print(f"  {key}: {value:+.3f}")

    print("\n✅ Sentiment analysis working!")
    print("\nTo use FinBERT (more accurate):")
    print("  1. pip install transformers torch")
    print("  2. Set use_finbert=True")
    print("\nTo get real data:")
    print("  1. NewsAPI: https://newsapi.org")
    print("  2. Twitter API: https://developer.twitter.com")
    print("  3. Reddit PRAW: pip install praw")

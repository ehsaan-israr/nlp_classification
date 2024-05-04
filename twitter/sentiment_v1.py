import string
from collections import defaultdict

import nltk
import matplotlib
import nltk
import re
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import random


# load data from
nltk.download('twitter_samples')
nltk.download('stopwords')

# print(stopwords.words())


class TweetSentimentBasicModel:
    tokenizer = TweetTokenizer()
    stopwords = stopwords.words("english")
    stemmer = PorterStemmer()
    positive_words_frequency = defaultdict(int)
    negative_words_frequency = defaultdict(int)

    def __init__(self):
        pass

    @staticmethod
    def remove_hashtags_and_urls(tweet):
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)

        # remove hyperlinks
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)

        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)

        return tweet

    @classmethod
    def tokenize_and_lower(cls, tweet):
        tokens = cls.tokenizer.tokenize(tweet)
        return [token.lower() for token in tokens]

    @classmethod
    def remove_stopword_punctuation_and_stem(cls, tokens):
        processed_tokens = []
        for token in tokens:
            if token not in cls.stopwords and token not in string.punctuation:
                processed_tokens.append(cls.stemmer.stem(token))
        return list(set(processed_tokens))

    @classmethod
    def preprocess_tweet(cls, tweet):
        tweet = cls.remove_hashtags_and_urls(tweet)
        tokens = cls.tokenize_and_lower(tweet)
        tokens = cls.remove_stopword_punctuation_and_stem(tokens)
        return tokens

    @classmethod
    def build_frequency(cls, tweets):
        words_frequency_data = defaultdict(int)
        for tweet in tweets:
            tokens = cls.preprocess_tweet(tweet)
            for token in tokens:
                words_frequency_data[token] += 1
        return words_frequency_data

    @classmethod
    def train(cls):
        # build positive words frequency
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        cls.positive_words_frequency = cls.build_frequency(positive_tweets)

        # build negative words frequency
        negative_tweets = twitter_samples.strings('negative_tweets.json')
        cls.negative_words_frequency = cls.build_frequency(negative_tweets)

    @classmethod
    def predict_tweet(cls, tweet):
        tokens = cls.preprocess_tweet(tweet)

        # set positive_count count
        positive_count = 0
        for token in tokens:
            positive_count += cls.positive_words_frequency[token] or 0

        # set negative count
        negative_count = 0
        for token in tokens:
            negative_count += cls.negative_words_frequency[token] or 0

        if negative_count > positive_count:
            return 'Tweet is negative'
        elif positive_count > negative_count:
            return 'Tweet is positive'
        else:
            return 'Tweet is neutral'


obj = TweetSentimentBasicModel()
# obj.train()
# print(obj.predict_tweet('I am sad because i am not learning NLP'))
# print(obj.predict_tweet('I am happy because i am not learning NLP'))
# print(obj.predict_tweet('It\'s great'))
# print(obj.predict_tweet('I want to fuck you'))
# print(obj.predict_tweet('It\' fucking shit'))
# print(obj.predict_tweet('i want to have some money'))
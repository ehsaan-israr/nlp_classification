import re
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import nltk
import numpy
from nltk import TweetTokenizer, PorterStemmer
from nltk.corpus import twitter_samples, stopwords

# load data from
nltk.download('twitter_samples')
nltk.download('stopwords')


class TweetSentimentSigmoidModel:
    words_frequency = defaultdict(int)
    tokenizer = TweetTokenizer()
    stopwords = stopwords.words("english")
    stemmer = PorterStemmer()

    @staticmethod
    def remove_ignore_words(tweet):
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
        tweet = cls.remove_ignore_words(tweet)
        tokens = cls.tokenize_and_lower(tweet)
        tokens = cls.remove_stopword_punctuation_and_stem(tokens)
        return tokens

    @classmethod
    def build_frequency(cls, tweets, label):
        for tweet in tweets:
            tokens = cls.preprocess_tweet(tweet)
            for token in tokens:
                cls.words_frequency[(token, label)] += 1
        return cls.words_frequency

    @classmethod
    def train_using_frequency(cls):
        # build positive words frequency
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')

        cls.build_frequency(positive_tweets, label=1)
        cls.build_frequency(negative_tweets, label=0)

    @staticmethod
    def visualize_tweet_words(data=None, size=(20, 20), ):
        fig, ax = plt.subplots(figsize=size)
        positive_data_x = numpy.log([item[1] + 1 for item in data])
        negative_data_y = numpy.log([item[2] + 1 for item in data])
        ax.scatter(positive_data_x, negative_data_y)
        plt.xlabel("Positive words count")
        plt.ylabel("Negative words count")
        # Add the word as the label at the same position as you added the points just before
        for i in range(0, len(data)):
            ax.annotate(data[i][0], (positive_data_x[i], negative_data_y[i]), fontsize=4)
        ax.plot([0, 9], [0, 9], color='red')  # Plot the red line that divides the 2 areas.
        plt.show()

    @classmethod
    def predict_tweet(cls, tweet, parameter):
        tokens = cls.preprocess_tweet(tweet)

        # make a vector
        positive_count, negative_count = 0, 0
        data = []
        for token in tokens:
            positive_count += cls.words_frequency[(token, 1)] or 0
            negative_count += cls.words_frequency[(token, 0)] or 0
        data = [1, positive_count, negative_count]
        data = numpy.array(data)
        parameter = numpy.array(parameter)
        dot_product = numpy.dot(data, parameter)

        sigmoid_z = 1 / (1 + numpy.exp(-dot_product))
        print(f"tweet:'{tweet}', feature vector:{data} dot_product: {dot_product}, sigmoid_z: {sigmoid_z}")

        # draw on the plot
        # cls.visualize_tweet_words(data=data, size=(100,100))

        # calculate status
        if negative_count > positive_count:
            return 'Tweet is negative'
        elif positive_count > negative_count:
            return 'Tweet is positive'
        else:
            return 'Tweet is neutral'


obj_v3 = TweetSentimentSigmoidModel()
obj_v3.train_using_frequency()
parameter = numpy.array([0.00003, 0.00150, -0.00120])
print('#################### TweetSentimentVectorModel ##################')
print(obj_v3.predict_tweet('I am sad because i am not learning NLP', parameter))
print(obj_v3.predict_tweet('I am happy because i am not learning NLP', parameter))
print(obj_v3.predict_tweet('It\'s great', parameter))
print(obj_v3.predict_tweet('I want to fuck you', parameter))
print(obj_v3.predict_tweet('It\' fucking shit', parameter))
print(obj_v3.predict_tweet('i want to have some money', parameter))
print(obj_v3.predict_tweet('@ali and @aslam are tuning a great ai model', parameter))
print(obj_v3.predict_tweet('pakistan is politically unstable country', parameter))

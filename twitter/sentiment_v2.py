from collections import defaultdict

import matplotlib.pyplot as plt
import numpy
from nltk.corpus import twitter_samples

from twitter.sentiment_v1 import TweetSentimentBasicModel


class TweetSentimentVectorModel(TweetSentimentBasicModel):
    words_frequency = defaultdict(int)

    @classmethod
    def build_frequency_v2(cls, tweets, labels):
        for tweet, label in zip(tweets, labels):
            tokens = cls.preprocess_tweet(tweet)
            for token in tokens:
                cls.words_frequency[(token, label)] += 1
        return cls.words_frequency

    @classmethod
    def train(cls):
        # build positive words frequency
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')

        cls.words_frequency = cls.build_frequency_v2(positive_tweets, labels=list(numpy.ones(len(positive_tweets))))
        cls.words_frequency = cls.build_frequency_v2(negative_tweets, labels=list(numpy.zeros(len(negative_tweets))))

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
    def predict_tweet(cls, tweet):
        tokens = cls.preprocess_tweet(tweet)

        # make a vector
        positive_count, negative_count = 0, 0
        data = []
        for token in tokens:
            positive_count += cls.words_frequency[(token, 1)] or 0
            negative_count += cls.words_frequency[(token, 0)] or 0
            data.append((token, positive_count, negative_count))

        # print tweet data
        print(data)

        # draw on the plot
        # cls.visualize_tweet_words(data=data, size=(100,100))

        # calculate status
        if negative_count > positive_count:
            return 'Tweet is negative'
        elif positive_count > negative_count:
            return 'Tweet is positive'
        else:
            return 'Tweet is neutral'


obj_v2 = TweetSentimentVectorModel()
obj_v2.train()
print('#################### TweetSentimentVectorModel ##################')
print(obj_v2.predict_tweet('I am sad because i am not learning NLP'))
print(obj_v2.predict_tweet('I am happy because i am not learning NLP'))
print(obj_v2.predict_tweet('It\'s great'))
print(obj_v2.predict_tweet('I want to fuck you'))
print(obj_v2.predict_tweet('It\' fucking shit'))
print(obj_v2.predict_tweet('i want to have some money'))
print(obj_v2.predict_tweet('@ali and @aslam are tuning a great ai model'))

#!/usr/bin/env python

"""
Analysing tweets
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import json
from main.sentiment import sentiment_mod as senti

# Add your own
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""


class listener(StreamListener):
    def on_data(self, raw_data):
        try:
            all_data = json.loads(raw_data)
            tweet = all_data["text"]
            sentiment_value, confidence = senti.sentiment(tweet)
            if confidence * 100 >= 80:
                output = open("twitter_out.txt", "a")
                output.write(sentiment_value)
                output.write(":\t")
                output.write(tweet)
                output.write("\n")
                output.close()
            # print(tweet, "\n")
            return True
        except BaseException:
            return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["movie"])

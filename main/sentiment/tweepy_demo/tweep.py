#!/usr/bin/env python

"""
tweepy(Twitter API) demo
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import time


# Add your own
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""


class listener(StreamListener):
    def on_data(self, raw_data):
        try:
            # print(raw_data)
            tweet = raw_data.split(",\"text\":")[1].split(",\"source\"")[0]
            print(tweet)
            save_time = str(time.time()) + "::" + tweet
            save_file = open('tweetDB.csv', 'a')
            save_file.write(save_time)
            save_file.write("\n")
            save_file.close()
            return True
        except BaseException:
            print("Failed")

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["<anything: noun/verb/adverb/...>"])

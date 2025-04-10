import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import TweetTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from random import randint, shuffle, seed
#from word2vec import Word2Vec

import dataloading
from evaluation import record_metrics

#word_embeddings = fasttext.load_model("../cc.en.300.bin")
#word_embeddings = Word2Vec(210325)

class FeatureExtractor():
    NUM_TOKEN_FEATURES = 500
    NUM_SENTI_FEATURES = 4
    NUM_WORD_EMBEDDING_FEATURES = 300
    NUM_LENGTH_FEATURE = 1

    FEATURE_START_TOKENS = 0
    FEATURE_START_SENTI = FEATURE_START_TOKENS + NUM_TOKEN_FEATURES
    FEATURE_START_WORD_EMBEDDING = FEATURE_START_SENTI + NUM_SENTI_FEATURES
    FEATURE_START_LENGTH = FEATURE_START_WORD_EMBEDDING + NUM_WORD_EMBEDDING_FEATURES

    NUM_FEATURES = FEATURE_START_LENGTH + NUM_LENGTH_FEATURE

    def __init__(self):
        self.__training = True
        self.lexicon = dict[str, int]()
        self.sentiment = SentimentIntensityAnalyzer()

        self.tfidf_transformer = TfidfTransformer()

    def make_feature(self, tweets: pd.Series, training: bool):
        self.__training = training

        tokenised_tweets: list[list[str]] = []

        tokeniser = TweetTokenizer(preserve_case = True)
        for tweet in tweets:
            tokens = tokeniser.tokenize(str(tweet))

            tokenised_tweets.append(tokens)

        if self.__training:
            all_tokens = Counter[str]()
            self.avg_length = 0
            for tweet in tokenised_tweets:
                self.avg_length += len(tweet)
                all_tokens.update(token.lower() for token in tweet)
            self.avg_length /= len(tokenised_tweets)

            for token, _ in all_tokens.most_common(self.NUM_TOKEN_FEATURES):
                if token not in self.lexicon:
                    self.lexicon[token.lower()] = len(self.lexicon)

        tf_idf_mat = np.zeros((len(tokenised_tweets), self.NUM_TOKEN_FEATURES))
        for i, tweet in enumerate(tokenised_tweets):
            for token in tweet:
                token = token.lower()
                if token in self.lexicon:
                    tf_idf_mat[i,self.lexicon[token]] += 1

        if self.__training:
            self.tfidf_transformer.fit(tf_idf_mat)

        tf_idf_mat = self.tfidf_transformer.transform(tf_idf_mat).todense()

        feature_mat = np.zeros((len(tweets), self.NUM_FEATURES))
        for i, (tweet_tokenised, tweet_untokenised) in enumerate(zip(tokenised_tweets, tweets)):
            #for token in tweet:
            #    if token in self.lexicon:
            #        feature_mat[i,self.FEATURE_START_TOKENS+self.lexicon[token.lower()]] = 1
#
            #        feature_mat[i,self.FEATURE_START_WORD_EMBEDDING:self.FEATURE_START_WORD_EMBEDDING+self.NUM_WORD_EMBEDDING_FEATURES] += word_embeddings.get_word_vector(token)
            #feature_mat[i,self.FEATURE_START_TOKENS:self.FEATURE_START_TOKENS+self.NUM_TOKEN_FEATURES] = tf_idf_mat[i]

            #feature_mat[i,self.FEATURE_START_WORD_EMBEDDING:self.FEATURE_START_WORD_EMBEDDING+self.NUM_WORD_EMBEDDING_FEATURES] *= 1.0 / len(tweet)

            #for token in tweet:
            #    if token.lower() not in self.lexicon:
            #        feature_mat[i,self.FEATURE_START_WORD_EMBEDDING:self.FEATURE_START_WORD_EMBEDDING+self.NUM_WORD_EMBEDDING_FEATURES] \
            #            = word_embeddings.get_word_embedding_or_random(token)
            #        break

            polarity = self.sentiment.polarity_scores(str(tweet_untokenised))
            feature_mat[i,self.FEATURE_START_SENTI+0] = polarity["pos"]
            feature_mat[i,self.FEATURE_START_SENTI+1] = polarity["neg"]
            feature_mat[i,self.FEATURE_START_SENTI+2] = polarity["neu"]
            feature_mat[i,self.FEATURE_START_SENTI+3] = polarity["compound"]

            feature_mat[i,self.FEATURE_START_LENGTH] = len(tweet_tokenised) / self.avg_length

        #if self.training:
        #    self.feature_mat_norm = feature_mat.sum(axis = 1) / self.NUM_TOKEN_FEATURES

        return feature_mat

dataloading.get_main_df()

training_set = pd.read_csv("datasets/train.En.csv")
test_set = pd.read_csv("datasets/task_A_En_test.csv")

training_tweets = list(training_set["tweet"])
#training_tweets.extend(training_set["rephrase"])

y_train_true = list(training_set["sarcastic"])

#for _ in range(300):
#    i = randint(0, len(training_tweets)-1)
#    if y_train_true[i] == 0:
#        del training_tweets[i]
#        del y_train_true[i]

#seed(12345)
#shuffle(training_tweets)

#y_train_true.extend(0 for i in range(len(training_set)))

#seed(12345)
#shuffle(y_train_true)


datasets = [
    ("main_set_only", dataloading.get_main_df()),
    ("full_set_hashes", dataloading.get_full_df(1)),
    ("full_set_no_hashes", dataloading.get_full_df(2))
]

for name, (train, test) in datasets:
    extractor = FeatureExtractor()
    X_train = extractor.make_feature(train["tweet"], True)
    y_train_true = train["sarcastic"]
    X_test = extractor.make_feature(test["text"], False)
    y_test_true = test["sarcastic"]

    svm = SVC(class_weight = "balanced")
    svm.fit(X_train, y_train_true)

    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    record_metrics(y_test_true, y_test_pred, name, display = True)

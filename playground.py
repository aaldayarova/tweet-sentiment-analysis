#Import tweets from NLTK that contain the following:
#'negative_tweets.json' = 5000 tweets with negative sentiment
#'positive_tweets.json' = 5000 tweets with positive sentiment
#'tweets.20150430-223406.json' = 20000 tweets with no sentiment
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re, string, random

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub('(@[A-Za-z0-9_]+)','', token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    #Create variables for different sentiment tweets
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')

    #Tokenizing tweets
    pos_tweets_tokenized = twitter_samples.tokenized('positive_tweets.json')
    neg_tweets_tokenized = twitter_samples.tokenized('negative_tweets.json')

    #Normalizing tweets (a.k.a. converting different forms of the same word to their canonical form)
    #'Stemming' (a.k.a. removing affixes) and 'lemmatization' are two normalization techniques
    #The lemmatization algorithm analyzes the structure of the word and its context to convert it to a normalized form
    #^Taken from https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

    #Removing noise (e.g. URLs, punctuation, in replies) AND normalizing
    stop_words = stopwords.words('english')

    pos_clean_tweets_tokenized = []
    neg_clean_tweets_tokenized = []

    for tokens in pos_tweets_tokenized:
        pos_clean_tweets_tokenized.append(remove_noise(tokens, stop_words))

    for tokens in neg_tweets_tokenized:
        neg_clean_tweets_tokenized.append(remove_noise(tokens, stop_words))

    #Performing analysis on cleaned positive data (most frequently, word frequency analysis)
    all_pos_words = get_all_words(pos_clean_tweets_tokenized)

    freq_dist_pos = FreqDist(all_pos_words)

    #Preparing data for sentiment analysis
    pos_tokens_for_model = get_tweets_for_model(pos_clean_tweets_tokenized)
    neg_tokens_for_model = get_tweets_for_model(neg_clean_tweets_tokenized)

    #Splitting data for training and testing
    pos_dataset = [(tweet_dict, 'Positive') for tweet_dict in pos_tokens_for_model]
    neg_dataset = [(tweet_dict, 'Negative') for tweet_dict in neg_tokens_for_model]
    dataset = pos_dataset + neg_dataset
    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    #Building and testing the model
    classifier = NaiveBayesClassifier.train(train_data)

    #print('Accuracy is: ', classify.accuracy(classifier, test_data))
    #print(classifier.show_most_informative_features(10))

    #Check how the model performs on a random tweet
    custom_tweet = "Incredible job yesterday!"
    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    #print(classifier.classify(dict([token, True] for token in custom_tokens)))

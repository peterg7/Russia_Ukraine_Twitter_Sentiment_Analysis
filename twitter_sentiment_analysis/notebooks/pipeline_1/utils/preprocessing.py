
# Import dependencies
import re
import multiprocessing

from shared_imports import np, pd, ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Prep nltk library
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def cleanAndFilter(raw_tweets_df, clean_tweet_regexs=[], clean_hashtag_regexs=[], column_mappings={}, 
                        filter_words=[], signals=[], **kwargs):
    # Rename columns
    if column_mappings:
        tweets_df = raw_tweets_df[list(column_mappings.keys())].rename(columns=column_mappings)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    ('No provided mapping for columns. Assuming valid format.')))
        tweets_df = raw_tweets_df.copy()

    # Drop duplicate tweets
    tweets_df = tweets_df.drop_duplicates(subset='tweet', keep='first')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    # Setup tweet cleaning
    if not clean_tweet_regexs:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    ('No provided patterns for tweet cleaning. Will still case-standardize, ' +
                                    'remove stopwords, strip, and stem values.')))
    
    def cleanTweet(tweet):
        tweet = tweet.lower()
        for pattern in clean_tweet_regexs:
            tweet = re.sub(pattern, ' ', tweet)
        tweet = ' '.join([w for w in tweet.split() if len(w) > 1])
        
        trimmed_lemma_words = [lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(tweet) 
                                    if x not in stop_words]
        clean_tweet = ' '.join(trimmed_lemma_words)
        
        return [lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) 
                    for x in nltk.wordpunct_tokenize(clean_tweet) if x not in stop_words]

    # Setup hashtag cleaning
    if not clean_hashtag_regexs:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    ('No provided patterns for hashtag cleaning. ' +
                                    'Will still case-standardize and strip values.')))

    def cleanHashtags(hashtags):
        if hashtags:
            hashtags = hashtags.lower()
            for pattern in clean_hashtag_regexs:
                hashtags = re.sub(pattern, ' ', hashtags)
            hashtags = hashtags.strip() 
        return hashtags


    # Clean tweets
    tweets_df['clean_tweet_words'] = tweets_df['tweet'].apply(lambda x: cleanTweet(x))
    tweets_df['clean_tweet'] = tweets_df['clean_tweet_words'].apply(lambda x:' '.join(x))

    # Clean hashtags
    tweets_df["hashtags"] = tweets_df["hashtags"].astype(str)
    tweets_df["hashtags"] = tweets_df["hashtags"].apply(lambda x: cleanHashtags(x))

    # Convert date to datetime and extract month/year
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    tweets_df['day'] = tweets_df['date'].dt.day
    tweets_df['month'] = tweets_df['date'].dt.month
    # tweets_df['year'] = tweets_df['date'].dt.year

    # Filter rows which don't have any clean words
    tweets_df.replace([], np.nan, inplace=True)
    tweets_df.dropna(subset=['clean_tweet_words'], inplace=True)

    if (filter_words):
        # Remove all tweets which do not have the provided target words
        keywords_str = '|'.join(filter_words)
        filtered_tweets_df = tweets_df.copy()
        filtered_tweets_df = filtered_tweets_df[filtered_tweets_df["clean_tweet"].str.contains(keywords_str)]
        
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    ('No provided filter words. Will process entire dataset.')))
        filtered_tweets_df = tweets_df.copy()
    
    return signals, (tweets_df, filtered_tweets_df)



def buildWordVectors(tweets_df, signals=[], **kwargs):
    # Restructure the `clean_text` column into list of list of words
    row_sentences = [row for row in tweets_df["clean_tweet_words"]]

    # Detect common phrases (bigrams) from a list of sentences
    if (phrases_params := kwargs.get('phrases_params')):
        phrases = Phrases(row_sentences, **phrases_params)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    "No provided arguments for `Phrases` object. Using gensim's default values."))
        phrases = Phrases(row_sentences)
        
    bigram = Phraser(phrases)
    sentences = bigram[row_sentences]
    
    # Initialize vector model
    num_cores = multiprocessing.cpu_count()-1
    if (word_vec_params := kwargs.get('word_vec_params')):
        if ('workers' in word_vec_params):
            workers = word_vec_params.get('workers')
            if not workers:
                word_vec_params['workers'] = num_cores
            word_vec_model = Word2Vec(**word_vec_params)
        else:
            word_vec_model = Word2Vec(workers=num_cores, **word_vec_params)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    "No provided arguments for `Word2Vec` object. Using gensim's default values."))
        word_vec_model = Word2Vec(workers=num_cores)
    
    # Establish dataset for the vector model
    if (word_vec_vocab := kwargs.get('word_vec_vocab')):
        word_vec_model.build_vocab(sentences, **word_vec_vocab)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    "No provided arguments for `Word2Vec.build_vocab()`. Using gensim's default values."))
        word_vec_model.build_vocab(sentences)

    # Train the model
    if (word_vec_train := kwargs.get('word_vec_train')):
        word_vec_model.train(sentences, total_examples=word_vec_model.corpus_count, **word_vec_train)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                    "No provided arguments for `Word2Vec.train()`. Using gensim's default values."))
        word_vec_model.train(sentences, total_examples=word_vec_model.corpus_count)
    
    return signals, word_vec_model.wv

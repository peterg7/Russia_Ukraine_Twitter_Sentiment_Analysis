#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis Pipeline #1
# *Refer to `notebooks/README.md` for an explanation of the various pipelines*

# ## Import dependencies



# Built-in
import re
import sys
import os
import json
import multiprocessing
from collections import defaultdict

# Importing datasets
import opendatasets as od

# Data manipulation
import pandas as pd
import numpy as np
from joblib import dump, load

# Graphing/Visualizing
from IPython.display import display




# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB




# NLP
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Prep nltk library
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')




# User-defined (a bit jank...)
sys.path.append("../utils") # Adds higher directory for ControlSignal
from ControlSignal import ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS, processSignals
from ConfigParser import validateConfig
from Grapher import graphWordDist, graphTweetDist
sys.path.remove('../utils')


# ## Build `extract` function



def extract(import_path, import_dest, **kwargs):
    signals=[]
    if kwargs.get('new_dataset'): 
        # Requesting new data
        dest_dir = os.path.dirname(import_dest)
        
        # Check for existing dataset
        if os.path.isfile(import_dest):
            print('Found existing file:', import_dest)
            user_input = input('Remove? (y/n)')
            if user_input != 'y': # add a version suffix to newly imported file
                existing_files = [f for f in os.listdir(dest_dir) if f.endswith('.csv')]
                parts = import_dest.split('/')
                ext_index = parts[-1].index('.')
                parts[-1] = f"{parts[-1][:ext_index]}_({len(existing_files)}){parts[-1][ext_index:]}"
                import_dest = os.path.join(*(parts))
            else:
                try:
                    os.remove(import_dest)
                except OSError as e:
                    signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_LOCATION, f'Could not delete existing file. Received error {str(e)}'))

        # Download dataset
        od.download(import_path, data_dir=dest_dir)

        # Collect downloaded dataset
        ## ASSUMING KAGGLE DOWNLOAD!!
        dataset_id = od.utils.kaggle_direct.get_kaggle_dataset_id(import_path) # in form 'username/dataset_name'
        data_import_dir = os.path.join(dest_dir, dataset_id.split('/')[1]) 

        imported_filename = next(f for f in os.listdir(data_import_dir) if f.endswith('.csv'))

        if not imported_filename:
            print('Error importing data. File was either not downloaded or moved')
            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Failed importing data. File was either moved or not downloaded. Searched for .csv file in {data_import_dir}'))
            return signals, None

        # Move and rename file
        os.rename(os.path.join(data_import_dir, imported_filename), import_dest)

        # Remove temporary directory created when downloaded
        try:
            if os.listdir((data_import_dir)):
                signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Import directory [{data_import_dir}] already exists! Unable to move new data'))
                return signals, None
            os.rmdir(data_import_dir)
        except OSError as e:
            print('Could not delete import directory, got' + str(e))
            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Could not delete import directory [{data_import_dir}]. Exiting for safety.'))
            return signals, None


    # Import data
    raw_tweets_df = pd.read_csv(import_dest)

    # Look for existing models
    existing_models = {}
    if (word_vec_path := kwargs.get('word_vec')):
        existing_models['word_vec'] = Word2Vec.load(word_vec_path).wv
    if (kmeans_path := kwargs.get('kmeans')):
        existing_models['kmeans'] = load(kmeans_path)
    if (embeddings_path := kwargs.get('embeddings')):
        existing_models['embeddings'] = pd.read_csv(embeddings_path)
    if (vectorizer_path := kwargs.get('vectorizer')):
        existing_models['vectorizer'] = load(vectorizer_path)
    if (linear_svc_path := kwargs.get('linear_svc')):
        existing_models['linear_svc'] = (load(linear_svc_path), None) # Place holder for performance metrics
    if (multi_nb_path := kwargs.get('multi_nb')):
        existing_models['multi_nb'] = (load(multi_nb_path), None) # Place holder for performance metrics

    return signals, (raw_tweets_df, existing_models)


# ## Define helper function for `transform`
# Handles user input for cluster sentiment assignment



## Need user input to determine each cluster's sentiment ##

def setClusterSentiment(vectors, model, mapping, signals=[], display_terms=20):
    print(f'** Top {display_terms} Similar Word Vectors By Cluster **\n')

    def collectSamples(multiplier=0):
        word_vec_list = [vectors.similar_by_vector(model.cluster_centers_[x], 
                                                            topn=(display_terms * (multiplier+1)), 
                                                            restrict_vocab=None) for x in range(len(mapping))]
        
        cluster_values = np.array(list(zip(*[x[(display_terms * multiplier):] for x in word_vec_list])))

        # Collect terms spanning multiple clusters for deciphering
        term_freq, counts = np.unique([x[0] for x in np.vstack(cluster_values)], axis=0, return_counts=True)
        unique_terms = term_freq[counts == 1]

        # Separate unique from duplicate terms
        uniq_cluster_vals = defaultdict(lambda : np.full(len(cluster_values), np.nan, dtype=object))
        shared_cluster_vals = defaultdict(lambda : [0] * len(mapping))
        for iy, ix in np.ndindex(cluster_values.shape[:len(mapping)-1]):
            tmp = cluster_values[iy, ix]
            if tmp[0] in unique_terms:
                uniq_cluster_vals[ix][iy] = tuple(tmp)
            else:
                shared_cluster_vals[tmp[0]][ix] = tmp[1]

        max_uniq_in_cluster = max([len([x for x in l if not pd.isnull(x)]) for l in uniq_cluster_vals.values()])
        formatted_unique = np.array([np.pad(vals[~pd.isnull(vals)], 
                                        (0,max_uniq_in_cluster-np.count_nonzero(~pd.isnull(vals))), constant_values=None) 
                                        for vals in uniq_cluster_vals.values()], dtype=object).T

        cols = [f'Cluster {x}' for x in range(len(mapping))]

        print('Unique Terms from Clusters')
        unique_terms_df = pd.DataFrame([[x[0] if x else '' for x in y] for y in formatted_unique], columns=cols)
        display(unique_terms_df)

        print('\nDuplicate Terms from Clusters')
        duplicate_terms_df = pd.DataFrame.from_dict(shared_cluster_vals, orient='index', columns=cols)
        display(duplicate_terms_df)
    
    collectSamples()

    ## Get input
    
    print('\nLabel each cluster: -1 = negative, 0 = neutral, 1 = positive ("r" for new samples, "q" to exit)')
    cluster_sentiment_defs = []
    user_input = ''
    resets = 0
    while len(cluster_sentiment_defs) < len(mapping)-1 and user_input != 'q':
        
        user_input = input(f'Cluster {len(cluster_sentiment_defs)} value:')
        if user_input == 'q':
            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.USER_INPUT))
            return
        if user_input == 'r':
            print(f'\n\nGenerating next {display_terms} samples...\n')
            resets += 1
            collectSamples(resets)
            print('Current state:', cluster_sentiment_defs)
            print('Setting cluster:', len(cluster_sentiment_defs))
            continue
        try:
            value = int(user_input)
            if value in cluster_sentiment_defs or value not in range(-1, 2):
                print('Already used this sentiment or not in range (-1, 0, 1)')
                continue
            cluster_sentiment_defs.append(value)
            print(f'Set cluster {len(cluster_sentiment_defs)-1} to {value} ({mapping[value]})')
        except ValueError:
            print('Need a number in range [-1, 0, 1]. Press q to exit')
            
    cluster_sentiment_defs.append((set(mapping.keys()) - set(cluster_sentiment_defs)).pop())

    print(f'Set cluster {len(cluster_sentiment_defs)-1} to {cluster_sentiment_defs[-1]} ({mapping[cluster_sentiment_defs[-1]]})')
    return cluster_sentiment_defs


# ## Build `transform` function



def cleanAndFilter(raw_tweets_df, column_mappings={}, filter_words={}, **kwargs):
    # Rename columns
    tweets_df = raw_tweets_df[list(column_mappings.keys())].rename(columns=column_mappings) \
                    if column_mappings else raw_tweets_df.copy()

    # Drop duplicate tweets
    tweets_df = tweets_df.drop_duplicates(subset='tweet', keep='first')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()

    stop_words = stopwords.words("english")
    tweet_regexs = kwargs.get('cleen_tweet', [r'https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*',
                                                r'\$[a-zA-Z0-9]*',
                                                r'[^a-zA-Z\']',
                                                r'\@[a-zA-Z0-9]*'])
    def cleanTweet(tweet):
        tweet = tweet.lower()
        for pattern in tweet_regexs:
            tweet = re.sub(pattern, ' ', tweet)
        tweet = ' '.join([w for w in tweet.split() if len(w) > 1])
        
        trimmed_lemma_words = [lemma.lemmatize(x) for x in nltk.wordpunct_tokenize(tweet) 
                                    if x not in stop_words]
        clean_tweet = ' '.join(trimmed_lemma_words)
        
        return [lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB) 
                    for x in nltk.wordpunct_tokenize(clean_tweet) if x not in stop_words]

    hashtag_regexs = kwargs.get('clean_hashtag', [r'\$[a-zA-Z0-9]*', r'[^a-zA-Z\']'])
    def cleanHashtags(hashtags):
        if hashtags:
            hashtags = hashtags.lower()
            for pattern in hashtag_regexs:
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

    if filter_words:
        # Remove all tweets which do not have the provided target words
        keywords_str = '|'.join(filter_words)
        filtered_tweets_df = tweets_df.copy()
        filtered_tweets_df = filtered_tweets_df[filtered_tweets_df["clean_tweet"].str.contains(keywords_str)]
        return tweets_df, filtered_tweets_df
        
    return tweets_df, pd.DataFrame([]) 




def buildWordVectors(tweets_df, progress_per=50000, epohcs=30, **kwargs):
    # Restructure the `clean_text` column
    row_sentences = [row for row in tweets_df["clean_tweet_words"]]

    # Detect common phrases (bigrams) from a list of sentences
    phrases = Phrases(row_sentences, min_count=1, progress_per=50000)
    bigram = Phraser(phrases)
    sentences = bigram[row_sentences]
    
    # Initialize vector model
    if (word_vec_params := kwargs.get('word_vec_args')):
        word_vec_model = Word2Vec(**word_vec_params)
        word_vec_model.build_vocab(sentences, progress_per=progress_per)
        word_vec_model.train(sentences, total_examples=word_vec_model.corpus_count, 
                                epochs=epohcs, report_delay=1)
    else:
        word_vec_model = Word2Vec(vector_size=300, 
                                window=5, 
                                min_count=4, 
                                workers=multiprocessing.cpu_count()-1,
                                negative=20, 
                                sample=1e-5, 
                                alpha=0.03, 
                                min_alpha=0.007,  
                                seed= 42)

        # Establish dataset for the vector model
        word_vec_model.build_vocab(sentences, progress_per=50000)

        # Train the model
        word_vec_model.train(sentences, total_examples=word_vec_model.corpus_count, 
                                epochs=30, report_delay=1)
    return word_vec_model.wv


def buildWordEmbeddings(word_vectors, model, sentiment_defs, sentiment_map):
     # Create a DataFrame of words with their embeddings and cluster values
    words_df = pd.DataFrame(word_vectors.index_to_key)
    words_df.columns = ['words']
    words_df['vectors'] = words_df.words.apply(lambda x: word_vectors[f'{x}'])
    words_df['cluster'] = words_df.vectors.apply(lambda x: model.predict([np.array(x)]))
    words_df.cluster = words_df.cluster.apply(lambda x: x[0])

    
    words_df['cluster_value'] = [sentiment_defs[i] for i in words_df.cluster]

    # Calculate proximity of words in each vector
    calc_vector_nearness = lambda x: 1 / (model.transform([x.vectors]).min())
    words_df['closeness_score'] = words_df.apply(calc_vector_nearness, axis=1)
    words_df['sentiment_coeff'] = words_df.closeness_score * words_df.cluster_value

    # Map sentiment encodings
    words_df["sentiment"] = words_df["cluster_value"].map(sentiment_map)
    return words_df

def peekSentimentDistrib(tweets_df):
    print('\nCalculated Sentiment Distribution:')
    display(tweets_df['sentiment'].value_counts())
    user_input = input('Distribution okay? (y/n) ')
    if user_input != 'y':
        return False
    return True




def transform(raw_tweets_df, sentiment_map, column_mappings={}, 
                                filter_words=[], existing_models={}, **kwargs):
    signals = []
    clean_tweets, filtered_tweets = cleanAndFilter(raw_tweets_df=raw_tweets_df, 
                                                    column_mappings=column_mappings, 
                                                    filter_words=filter_words, 
                                                    kwargs=kwargs)

    if (existing_word_vec := existing_models.get('word_vec')):
        word_vectors = existing_word_vec
    else:
        word_vectors = buildWordVectors(clean_tweets, kwargs)
    
    if (existing_kmeans := existing_models.get('kmeans')):
        cluster_model = existing_kmeans
    else:
        # Build KMeans model to cluster words into positive, negative, and neutral clusters
        if (kmeans_params := kwargs.get('kmeans_args')):
            cluster_model = KMeans(**kmeans_params)
        else:
            cluster_model = KMeans(n_clusters=3, max_iter=1000, 
                                    random_state=42, n_init=50)
        cluster_model = cluster_model.fit(X=word_vectors.vectors.astype('double'))
                        
    ############# Get Input ###############
    cluster_sentiment_defs = setClusterSentiment(vectors=word_vectors, 
                                                    model=cluster_model, 
                                                    mapping=sentiment_map, 
                                                    signals=signals,
                                                    display_terms=kwargs.get('display_terms'))
    if not cluster_sentiment_defs:
        return signals, None
    print('\nApplying sentiment mapping...')
    #######################################

    if (existing_embeddings := kwargs.get('embeddings')):
        words_df = existing_embeddings
    else:
        words_df = buildWordEmbeddings(word_vectors=word_vectors, 
                                        model=cluster_model, 
                                        sentiment_defs=cluster_sentiment_defs, 
                                        sentiment_map=sentiment_map)
        
    # Get the sentiment for the entire tweet
    threshold = kwargs.get('sentiment_threshold', 0.15)
    words_cluster_dict = dict(zip(words_df.words, words_df.cluster_value))
    def getSentiment(row):
        total, count = 0, 0
        test = row["clean_tweet_words"]
        for t in test:
            total += int(words_cluster_dict.get(t, 0))
            count += 1 
            
        avg = total / count
        return -1 if (avg < -threshold) else 1 if (avg > threshold) else 0

    # Add sentiment column (integer values)
    filtered_tweets["sentiment_val"] = filtered_tweets.apply(getSentiment, axis=1)
    # Map integer sentiment to word value
    filtered_tweets["sentiment"] = filtered_tweets["sentiment_val"].map(sentiment_map)

    # Confirm sentiment spread with user
    if not peekSentimentDistrib(filtered_tweets):
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.USER_INPUT, f'Distribution was unsatisfactory.'))
        return signals, None
    return signals, (filtered_tweets, words_df, cluster_sentiment_defs)


# ## Build `model` function



def testModel(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)

    # Build confusion matrix to evaluate the model results
    confusion = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))

    # Get classification report
    classification = classification_report(y_test, y_pred, labels=np.unique(y_pred))

    # Use score method to get accuracy of model
    acc_score = model.score(X_test, y_test)

    return {
        'confusion': confusion,
        'classification': classification,
        'acc_score': acc_score,
    }




def model(sentiment_df, existing_models={}, **kwargs):
    signals = []
    # Convert each sentiment to df (no need to worry about memory crash, small dataset)
    pos_df = sentiment_df[sentiment_df["sentiment"]=="positive"]
    neg_df = sentiment_df[sentiment_df["sentiment"]=="negative"]
    neu_df = sentiment_df[sentiment_df["sentiment"]=="neutral"]

    # Combine all sentiments in one df
    sentiments_df_list = [pos_df, neg_df, neu_df] 
    agg_sentiment_df = pd.concat(sentiments_df_list)

    # Split the data to training, testing, and validation data 
    test_size = kwargs.get('test_size', 0.2)
    train_test_df, _ = train_test_split(agg_sentiment_df, test_size=test_size, random_state=10)

    X = train_test_df['clean_tweet']
    y = train_test_df['sentiment_val']

    # Split the dataset set into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Instantiate TfidfVectorizer 
    if (existing_vectorizer := existing_models.get('vectorizer')):
        vectorizer = existing_vectorizer
    else: 
        if (vectorizer_params := kwargs.get('vectorizer_args')):
            vectorizer = TfidfVectorizer(**vectorizer_params)
        else:
            vectorizer = TfidfVectorizer(min_df=3,
                                            sublinear_tf=True,
                                            ngram_range=(1,2),
                                            stop_words='english')

    # Fit vectorizer
    X_train_tf = vectorizer.fit_transform(X_train.reset_index()["clean_tweet"]).toarray()
    X_test_tf = vectorizer.transform(X_test.reset_index()["clean_tweet"]).toarray()

    # Store features from the vectors
    feature_names = vectorizer.get_feature_names_out()

    # Create a dict of Sentiment_val: sentiments to use with the confusion matrix
    sentiment_id_df = agg_sentiment_df[['sentiment', 'sentiment_val']].drop_duplicates() \
                                                                        .sort_values('sentiment_val')
    sentiment_to_id = dict(sentiment_id_df.values)

    ## LinearSVC ##

    if (existing_linear_svc := existing_models.get('linear_svc')):
        # (NOTE: Must use same vectorizer from its fitting)
        linearSVC = existing_linear_svc
    else:
        # Instantiate the model
        linearSVC = LinearSVC()

        # Fit the model
        linearSVC.fit(X_train_tf, y_train)

    svc_performance = testModel(model=linearSVC, X_test=X_test_tf, y_test=y_test)

    ## MultinomialNB ##

    if (existing_multi_nb := existing_models.get('multi_nb')):
        # (NOTE: Must use same vectorizer from its fitting)
        multiNB = existing_multi_nb
    else:
        # Instantiate the model
        multiNB = MultinomialNB()

        # Fit the model
        multiNB.fit(X_train_tf, y_train)

    nb_performance = testModel(model=multiNB, X_test=X_test_tf, y_test=y_test)

    return signals, {
                'linear_svc': (linearSVC, svc_performance),
                'multi_nb': (multiNB, nb_performance),
                'vectorizer': vectorizer,
                'features': feature_names,
                'sentiment_id': sentiment_id_df,
                'sentiment_to': sentiment_to_id
            }


# ## Build `load` function



def load(transform_df, built_models={}, destinations={}, config=None):
    signals = []
    # Export the sentiment dataframe
    if (transform_dest := destinations.get('transform')):
        transform_df.to_csv(transform_dest)

    if (config_loc := destinations.get('config')):
        with open(config_loc, 'w') as f:
            json.dump(config, f)
    
    # Pair up models with their export destination 
    model_exports = { name: path for name, path in destinations.items() 
                                            if (name in built_models and built_models[name]) }
    for name, path in model_exports.items():
        if name == 'word_vec':
            built_models[name].save(path)
        elif name == 'embeddings':
            built_models[name].to_csv(path)
        elif isinstance(built_models[name], tuple): # Check for model with associated performance metrics
            dump(built_models[name][0], path)
        else:
            dump(built_models[name], path)
            
    # Save current notebook for import
    if (notebook_dest := destinations.get('notebook')):
        
        get_ipython().system('jupyter nbconvert --output {notebook_dest} --to script pipeline_1.ipynb')

        # Get rid of excess
        with open(notebook_dest + '.py', 'r+') as fp:
            lines = fp.readlines()
            fp.seek(0)
            fp.truncate()
            cell_markers = set([])
            term_index = len(lines) - 1
            for i, line in enumerate(lines):

#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis Pipeline #1
# *Refer to `notebooks/README.md` for an explanation of the various pipelines*

# ## Import dependencies



# Built-in
import re
import os
import json
import multiprocessing
from collections import defaultdict
from datetime import datetime

# Importing datasets
import opendatasets as od

# Data manipulation
import pandas as pd
import numpy as np
from joblib import dump, load

# Graphing/Visualizing
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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


# ## Define Parameters



DATA_IMPORT_PARAMS = {
    'data_path': 'https://www.kaggle.com/datasets/gpreda/slava-ukraini-tweets{version}',
    # 'version': 15,
    'local_path': '../../data/slava_ukraini_tweets{version}.csv'
}

CONTROL_PARAMS = {
    'build_models': True,
    'display_terms': 25,
    'config': './config/config_{timestamp}.json',
    'nb_export': '../pipeline_2/pipeline_1'
}

EXTRACT_ARGS = {
    'new_data': False,
    'clean_tweet': [],
    'clean_hashtag': [],
    'column_mappings': {
        'date': 'date',
        'user_name': 'username',
        'retweets': 'retweets',
        'text': 'tweet',
        'hashtags': 'hashtags'
    },
    'filter_words': ['ukraine', 'russia', 'zelensky'], # Only process tweets containing these words
    'save_transform': './data/transformed/slava_ukraine_sentiment_transform_{timestamp}.csv'
}

TRANSFORM_ARGS = {
    'load_word_vec': '',
    'save_word_vec': './models/slava_word_vec_{timestamp}.model',
    'word_vec_args': {},
    'load_kmeans': '',
    'save_kmeans': './models/slava_kmeans_{timestamp}.joblib',
    'kmeans_args': {},
    'display_terms': 25,
    'load_embeddings': '',
    'save_embeddings': './data/embeddings/slava_words_{timestamp}.csv',
    'sentiment_threshold': 0.15,
    'sentiment_map': { 
        -1: "negative",
        0: "neutral", 
        1: "positive"
    }
}

MODEL_ARGS = {
    'load_vectorizer': '',
    'save_vectorizer': './models/slava_vectorizer_{timestamp}.joblib',
    'vectorizer_args': {},
    'load_svc': '',
    'save_svc': './models/slava_linearSVC_{timestamp}.joblib',
    'load_nb': '',
    'save_nb': './models/slava_multinomialNB_{timestamp}.joblib'
}


# ## Helper functions



# Helper function to add suffix to file path
def addPathSuffix(path, suffix):
    parts = path.split('/')
    ext_index = parts[-1].index('.')
    parts[-1] = f"{parts[-1][:ext_index]}_{suffix}{parts[-1][ext_index:]}"
    return os.path.join(*(parts))


# Helper function to add current timestamps to file paths
def formatParams(params, timestamp=None):
    if not timestamp:
        timestamp = re.sub(':', '-', datetime.utcnow().isoformat('_', timespec='seconds'))
    
    formatted_params = dict(params)
    for key, val in params.items():
        if isinstance(val, str):
            if 'timestamp' in val:
                formatted_params[key] = val.format(timestamp=timestamp)
            elif 'version' in val:
                version = params.get('version')
                if key == 'data_path':
                    formatted_params[key] = val.format(version=f"/versions/{version}") if version else val.format(version='')
                else:
                    formatted_params[key] = val.format(version=f"_{version}") if version else val.format(version='')

    return formatted_params, timestamp


# ## Build `extract` function



def extract(import_path, local_path, column_mappings={}, filter_words=[], **kargs):

    if kargs.get('new_data'): 
        # Requesting new data
        dest_dir = os.path.join(*local_path.split('/')[:-1])
        
        # Check for existing dataset
        if os.path.isfile(local_path):
            print('Found existing file:', local_path)
            user_input = input('Remove? (y/n)')
            if user_input != 'y':
                existing_files = [f for f in os.listdir(dest_dir) if f.endswith('.csv')]
                local_path = addPathSuffix(local_path, f'({len(existing_files)})')
            else:
                try:
                    os.remove(local_path)
                except OSError as e:
                    print('Could not delete existing file, got' + str(e))

        # Download dataset
        od.download(import_path, data_dir=dest_dir)

        # Collect downloaded dataset
        data_import_dir = os.path.join(dest_dir, import_path.split('/')[-1])
        imported_file = next(f for f in os.listdir(data_import_dir) if f.endswith('.csv'))

        if not imported_file:
            print('Error importing data. File was either not downloaded or moved')
            return

        # Move and rename file
        temp_import_loc = os.path.join(data_import_dir, imported_file)
        os.rename(temp_import_loc, local_path)

        # Remove temporary directory created when downloaded
        try:
            assert not os.listdir((data_import_dir))
            os.rmdir(data_import_dir)
        except OSError as e:
            print('Could not delete import directory, got' + str(e))
            return

    # Import data
    raw_tweets_df = pd.read_csv(local_path)
    
    # Rename columns
    tweets_df = raw_tweets_df[list(column_mappings.keys())].rename(columns=column_mappings) \
                    if column_mappings else raw_tweets_df.copy()

    # Drop duplicate tweets
    tweets_df = tweets_df.drop_duplicates(subset='tweet', keep='first')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()

    stop_words = stopwords.words("english")
    tweet_regexs = kargs.get('cleen_tweet', [r'https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*',
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

    hashtag_regexs = kargs.get('clean_hashtag', [r'\$[a-zA-Z0-9]*', r'[^a-zA-Z\']'])
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
    tweets_df['month'] = tweets_df['date'].dt.month
    tweets_df['year'] = tweets_df['date'].dt.year

    if filter_words:
        # Remove all tweets which do not have the provided target words
        keywords_str = '|'.join(filter_words)
        target_tweets_df = tweets_df.copy()
        target_tweets_df = target_tweets_df[target_tweets_df["clean_tweet"].str.contains(keywords_str)]
        return tweets_df, target_tweets_df
        
    return tweets_df, None


# ## Define helper function for `transform`
# Handles user input for cluster sentiment assignment



## Need user input to determine each cluster's sentiment ##

def setClusterSentiment(vectors, model, mapping, display_terms=20):
    
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
            print('Aborting...')
            return []
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



def transform(filtered_df, cumulative_df, sentiment_map, **kargs):

    # Preseve original df
    transform_tweets_df = filtered_df.copy()
    
    if (load_word_vec := kargs.get('load_word_vec')):
        # Load existing vectors if path provided
        word_vectors = Word2Vec.load(load_word_vec).wv

    else:
        # Restructure the `clean_text` column
        row_sentences = [row for row in cumulative_df["clean_tweet_words"]]

        # Detect common phrases (bigrams) from a list of sentences
        phrases = Phrases(row_sentences, min_count=1, progress_per=50000)
        bigram = Phraser(phrases)
        sentences = bigram[row_sentences]
        
        # Initialize vector model
        if (word_vec_params := kargs.get('word_vec_args')):
            word_vec_model = Word2Vec(**word_vec_params)
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
        word_vec_model.train(sentences, 
                    total_examples=word_vec_model.corpus_count, 
                    epochs=30, 
                    report_delay=1)
        
        if (save_word_vec := kargs.get('save_word_vec')):
            # Store current word vector model
            word_vec_model.save(save_word_vec)

        word_vectors = word_vec_model.wv
    
    
    if (load_kmeans := kargs.get('load_kmeans')):
        cluster_model = load(load_kmeans)
    
    else:
        # Build KMeans model to cluster words into positive, negative, and neutral clusters
        if (kmeans_params := kargs.get('kmeans_args')):
            cluster_model = KMeans(**kmeans_params)
        else:
            cluster_model = KMeans(n_clusters=3, 
                                    max_iter=1000, 
                                    random_state=42, 
                                    n_init=50)
        cluster_model = cluster_model.fit(X=word_vectors.vectors.astype('double'))
                        

    ############# Get Input ###############
    
    cluster_sentiment_defs = setClusterSentiment(word_vectors, cluster_model, sentiment_map, kargs.get('display_terms'))

    if not cluster_sentiment_defs:
        return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])

    print('\nApplying sentiment mapping...')

    #######################################


    if (load_embeddings := kargs.get('load_embeddings')):
        # Load existing embeddings file
        words_df = pd.read_csv(load_embeddings)

    else:
        # Create a DataFrame of words with their embeddings and cluster values
        words_df = pd.DataFrame(word_vectors.index_to_key)
        words_df.columns = ['words']
        words_df['vectors'] = words_df.words.apply(lambda x: word_vectors[f'{x}'])
        words_df['cluster'] = words_df.vectors.apply(lambda x: cluster_model.predict([np.array(x)]))
        words_df.cluster = words_df.cluster.apply(lambda x: x[0])

        
        words_df['cluster_value'] = [cluster_sentiment_defs[i] for i in words_df.cluster]

        # Calculate proximity of words in each vector
        calc_vector_nearness = lambda x: 1 / (cluster_model.transform([x.vectors]).min())
        words_df['closeness_score'] = words_df.apply(calc_vector_nearness, axis=1)
        words_df['sentiment_coeff'] = words_df.closeness_score * words_df.cluster_value

        # Map sentiment encodings
        words_df["sentiment"] = words_df["cluster_value"].map(sentiment_map)

        if (save_embeddings := kargs.get('save_embeddings')):
            # Store current word embeddings
            words_df.to_csv(save_embeddings)


    if (save_kmeans := kargs.get('save_kmeans')):
        # Save clustering model if path provided
        dump(cluster_model, save_kmeans)
        

    # Get the sentiment for the entire tweet
    threshold = kargs.get('sentiment_threshold', 0.15)
    def getSentiment(row):
        total, count = 0, 0
        test = row["clean_tweet_words"]
        for t in test:
            total += int(words_cluster_dict.get(t, 0))
            # if score := words_cluster_dict.get(t):
            #     total += int(score)
            count += 1 
            
        avg = total / count
        return -1 if (avg < -threshold) else 1 if (avg > threshold) else 0


    # Create a dictionary of the word and its cluster value
    words_cluster_dict = dict(zip(words_df.words, words_df.cluster_value))

    # Add sentiment column (integer values)
    transform_tweets_df["sentiment_val"] = transform_tweets_df.apply(getSentiment,
                                                         axis=1)

    # Map integer sentiment to word value
    transform_tweets_df["sentiment"] = transform_tweets_df["sentiment_val"].map(sentiment_map)

    # Confirm sentiment spread with user
    print('\nCalculated Sentiment Distribution:')
    display(transform_tweets_df['sentiment'].value_counts())
    user_input = input('Distribution okay? (y/n) ')
    if user_input != 'y':
        print('Aborting...')
        return pd.DataFrame([]), pd.DataFrame([]), []


    return transform_tweets_df, words_df, cluster_sentiment_defs


# ## Build `model` function



def model(sentiment_df, test_size=0.2, **kargs):
    
    # Convert each sentiment to df (no need to worry about memory crash, small dataset)
    pos_df = sentiment_df[sentiment_df["sentiment"]=="positive"]
    neg_df = sentiment_df[sentiment_df["sentiment"]=="negative"]
    neu_df = sentiment_df[sentiment_df["sentiment"]=="neutral"]

    # Combine all sentiments in one df
    sentiments_df_list = [pos_df, neg_df, neu_df] 
    agg_sentiment_df = pd.concat(sentiments_df_list)

    # Split the data to training, testing, and validation data 
    train_test_df, _ = train_test_split(agg_sentiment_df, test_size=test_size, random_state=10)

    X = train_test_df['clean_tweet']
    y = train_test_df['sentiment_val']

    # Split the dataset set int0 training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Instantiate TfidfVectorizer 
    if (load_vectorizer := kargs.get('load_vectorizer')):
        vectorizer = load(load_vectorizer)
    else: 
        if (vectorizer_params := kargs.get('vectorizer_args')):
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

    if (load_svc := kargs.get('load_svc')):
        # Import previous linearSVC model (NOTE: Must use same vectorizer from its fitting)
        linearSVC = load(load_svc)

    else:
        # Instantiate the model
        linearSVC = LinearSVC(random_state=0)

        # Fit the model
        linearSVC.fit(X_train_tf, y_train)

        if (save_svc := kargs.get('save_svc')):
            dump(linearSVC, save_svc)

    # Predict
    svc_y_pred = linearSVC.predict(X_test_tf)

    # Build confusion matrix to evaluate the model results
    svc_conf_mat = confusion_matrix(y_test, svc_y_pred, labels=np.unique(svc_y_pred))

    # Get classification report
    svc_classification = classification_report(y_test, svc_y_pred, labels=np.unique(svc_y_pred))

    # Use score method to get accuracy of model
    svc_score = linearSVC.score(X_test_tf, y_test)

    ## MultinomialNB ##

    if (load_nb := kargs.get('load_nb')):
        # Import previous multinomialNB model (NOTE: Must use same vectorizer from its fitting)
        multiNB = load(load_nb)

    else:
        # Instantiate the model
        multiNB = MultinomialNB()

        # Fit the model
        multiNB.fit(X_train_tf, y_train)

        if (save_nb := kargs.get('save_nb')):
            dump(multiNB, save_nb)

    # Predict
    nb_y_pred = multiNB.predict(X_test_tf)

    # Build confusion matrix to evaluate the model results
    nb_conf_mat = confusion_matrix(y_test, nb_y_pred, labels=np.unique(nb_y_pred))

    # Get classification report
    nb_classification = classification_report(y_test, nb_y_pred, labels=np.unique(nb_y_pred))

    # Use score method to get accuracy of model
    nb_score = multiNB.score(X_test_tf, y_test)

    if (save_vectorizer := kargs.get('save_vectorizer')):
        dump(vectorizer, save_vectorizer)

    elif save_svc or save_nb:
        vector_out_match = re.search(r'\/[^/]*$', save_svc if save_svc else save_nb)
        if not vector_out_match.group():
            print('Error: could not extract default vectorizer output path')
        else:
            dump(vectorizer, re.sub(vector_out_match.group(), '/vectorizer.joblib', 
                    vector_out_match.string))

    return {
        'LinearSVC': {
            'model': linearSVC,
            'conf_mat': svc_conf_mat,
            'classification': svc_classification,
            'score': svc_score,
        },
        'MultinomialNB': {
            'model': multiNB,
            'conf_mat': nb_conf_mat,
            'classification': nb_classification,
            'score': nb_score,
        },
        'features': feature_names,
        'vectorizer': vectorizer,
        'sentiment_id': sentiment_id_df,
        'sentiment_to': sentiment_to_id
    }


# ## Build `load` function



def load(nb_dest):

    # Save current notebook for import by Pipeline 2
    get_ipython().system('jupyter nbconvert --output {nb_dest} --to script pipeline_1.ipynb')

    # Get rid of excess
    with open(nb_dest + '.py', 'r+') as fp:
        lines = fp.readlines()
        fp.seek(0)
        fp.truncate()
        cell_markers = []
        execute_start, execute_end = -1, -1
        for i, line in enumerate(lines):
            if '## Execute `pipeline`' in line:
                execute_start = i
            elif '## Visualizations' in line:
                execute_end = i
                cell_markers.append(i)
        
        exclude_list = list(range(execute_start, execute_end))
        exclude_list.extend(cell_markers)

        fp.writelines([l for i, l in enumerate(lines) if i not in set(exclude_list)])


# ## Build `pipeline` function from above processes



def pipeline1(import_path, import_params={}, extract_args={}, transform_args={}, 
                model_args={}, control_params={}):

    
    if not transform_args.get('sentiment_map'):
        transform_args['sentiment_map'] = { -1: "negative", 0: "neutral", 1: "positive" }

    print('Stage 1: Extracting...')

    config = defaultdict(dict)

    all_tweets_df, target_df = extract(import_path=import_path,
                                local_path=import_params.get('local_path', f"./{import_path.split('/')[-1]}.csv"),
                                **extract_args)

    print('Completed Stage 1.\n\nStage 2: Transforming...')
    transform_tweets_df, word_vecs, sentiment_defs = transform(filtered_df=target_df, 
                        cumulative_df=all_tweets_df,
                        **transform_args)

    model_dict = {}
    if not transform_tweets_df.empty:
        if sentiment_defs:
            config['sentiment_vals'] = {
                'value_mapping': transform_args['sentiment_map'],
                'cluster_mapping': sentiment_defs
            }

        print('Completed Stage 2.')

        if (save_transform := control_params.get('save_transform')):
            transform_tweets_df.to_csv(save_transform)

        if control_params.get('build_models'):
            print('\nStage 3: Modeling...')
            model_dict = model(sentiment_df=transform_tweets_df, **model_args)
            print('Completed Stage 3.')

        if (nb_dest := control_params.get('nb_export')):
            load(nb_dest)

        if (config_loc := control_params.get('config')):
            with open(config_loc, 'w') as f:
                json.dump(config, f)

    else:
        # Process aborted. Clean up...
        print('Cleaning...')
        save_params = ['save_word_vec', 'save_embeddings', 'save_kmeans']
        staged_paths = [transform_args[p] for p in save_params if transform_args.get(p)]
        for path in staged_paths:
            try:
                os.remove(path)
            except OSError as e:
                print('Could not delete file, got' + str(e))
        
    print('\n<done>')
    return transform_tweets_df, word_vecs, model_dict


# ## Visualizations



class GraphicProcessors:
    
    # Display a word cloud with the given text
    def generateWordcloud(text):
        words=' '.join([words for words in text])
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    # Shorthand functions for commonly used plots
    graphWordDistribution = lambda word_vecs: GraphicProcessors.graphDistribution(word_vecs, 'sentiment', 'Sentiment Distribution of Words')
    graphTweetDistribution = lambda tweet_df: GraphicProcessors.graphDistribution(tweet_df, 'sentiment', 'Sentiment Distribution of Tweets')

    # Make a pie chart from a dataframe's column distribution
    def graphDistribution(df, plot_col, title='', fig_size=(7, 7)):
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        colors = ["crimson", "steelblue", "mediumseagreen"]

        pie_df = df[plot_col].value_counts().reset_index()

        plt.pie(pie_df[plot_col],
                labels=pie_df["index"],
                radius=2,
                colors=colors,
                autopct="%1.1f%%")

        plt.axis('equal')
        if title:
            plt.title(title, fontsize=20)
        plt.show()
        return pie_df

    # Display a bar chart with the counts of values within a specific column
    def graphKeywordCounts(df, plot_col, count_col, keywords):
        # Inspect keyword sentiment
        pattern = '|'.join(keywords)
        keyword_sent_df = df[(df[plot_col].str.contains(pattern))]
        sns.countplot(x=keyword_sent_df[count_col]);

    # Shorthand functions for commonly used plots
    graphTop10Usernames = lambda tweets_df: GraphicProcessors.graphCounts(tweets_df, 'username', 'sentiment', 'Top 10 Highest Tweeting usernames', tweets_df['username'].value_counts().iloc[:10].index)
    graphTop10Hashtags = lambda tweets_df: GraphicProcessors.graphCounts(tweets_df, 'hashtags', 'sentiment', 'Top 10 Hashtags', tweets_df['hashtags'].value_counts().iloc[1:10].index, (15,10))

    # Display a more detailed bar chart from `graphKeywordSentiment`
    def graphCounts(df, x_col, hue_col=None, title='', order=None, plt_size=(10,8)):
        fig = plt.subplots(figsize=plt_size)
        if title:
            plt.title(title, fontsize=20)
        chart = sns.countplot(x=x_col, 
                                data=df, 
                                palette="Set2", 
                                hue=hue_col,
                                order=order)

        chart.set_xticklabels(chart.get_xticklabels(),
                                rotation=30, 
                                horizontalalignment='right')

    # Display a confusion matrix generated by a sklearn/tensorflow model
    def graphConfusionmatrix(conf_mat, sentiment_id_df):
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_mat, 
                    annot=True, 
                    fmt='d',
                    xticklabels=sentiment_id_df.sentiment.values, 
                    yticklabels=sentiment_id_df.sentiment_val.values)

        plt.ylabel('Actual')
        plt.xlabel('Predicted')

    # Graph a bar chart of the top portion of feature coefficients from a model
    def graphCoefficients(model, feature_names, top_features=20, fig_size=(15, 5)):

        coefficients_and_features = sorted(zip(model.coef_[0], feature_names)) 
        features_coef_df = pd.DataFrame(coefficients_and_features)
        features_coef_df.columns = 'coefficient','word'
        features_coef_df.sort_values(by='coefficient')

        num_features = len(feature_names)
        neg_coefficients = model.coef_[-1][:num_features]
        pos_coefficients = model.coef_[1][:num_features]
        top_pos_coefficients = np.argsort(pos_coefficients[pos_coefficients > 0])[-top_features:]
        top_neg_coefficients = np.argsort(pos_coefficients[neg_coefficients < 0])[:top_features]
        top_coefficients = np.hstack([top_neg_coefficients, top_pos_coefficients])
        total_coefficients = np.hstack([neg_coefficients, pos_coefficients])
        
        # create plot
        fig = plt.figure(figsize=fig_size)
        colors = ['red' if c < 0 else 'blue' for c in total_coefficients[top_coefficients]]
        feature_names = np.array(feature_names)

        plt.bar(np.arange(2 * top_features), total_coefficients[top_coefficients], color=colors)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        title="Positive and Negative Labels"
        plt.title(title)
        plt.show()


# ### Distributions



# GraphicProcessors.graphWordDistribution(word_vecs)




# GraphicProcessors.graphTweetDistribution(sentiment_df)




# GraphicProcessors.graphKeywordCounts(sentiment_df, 'clean_tweet', 'sentiment', ['russia'])


# ### Counts



# GraphicProcessors.graphTop10Usernames(sentiment_df)




# GraphicProcessors.graphTop10Hashtags(sentiment_df)


# ### Word Clouds



# Cloud for positive sentiment tweets
# GraphicProcessors.generateWordcloud(sentiment_df[sentiment_df['sentiment_val']==1]['clean_tweet'].values)




# Cloud for negative sentiment tweets
# GraphicProcessors.generateWordcloud(sentiment_df[sentiment_df['sentiment_val']==-1]['clean_tweet'].values)


# ### Confusion Matricies



# GraphicProcessors.graphConfusionmatrix(model_dict['LinearSVC']['conf_mat'], model_dict['sentiment_id'])


# ### Feature Coefficients



# GraphicProcessors.graphCoefficients(model_dict['LinearSVC']['model'], model_dict['features'])


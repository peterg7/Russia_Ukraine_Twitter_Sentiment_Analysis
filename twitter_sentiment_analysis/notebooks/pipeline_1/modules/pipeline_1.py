#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis Pipeline #1
# *Refer to `notebooks/README.md` for an explanation of the various pipelines*

# ## Import dependencies



from shared_imports import (ospath, oswalk, rmfile,
                            pd,
                            importData,
                            ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS, processSignals,
                            buildConfig,
                            Grapher)

from utils.preprocessing import *
from utils.model_assessment import *
from utils.sentiment_analysis import *




# Built-in
import json
import shutil
from collections import defaultdict
from operator import itemgetter

# Data manipulation
from joblib import dump, load




# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from gensim.models import Word2Vec


# ## Build `extract` function



def extract(import_path, import_dest, **kwargs):
    signals=[]
    print(f'Importing data from "{import_path}"...')

    _, raw_tweets_df = importData(import_loc=import_path, import_protocol=kwargs.get('import_path_protocol'),
                                  local_dest=import_dest, signals=signals, kwargs=kwargs)

    # Look for existing models/datasets
    existing_models = {}
    if (word_vec_path := kwargs.get('word_vec')):
        existing_models['word_vec'] = Word2Vec.load(word_vec_path).wv
    if (kmeans_path := kwargs.get('kmeans')):
        existing_models['kmeans'] = load(kmeans_path)
    if (embeddings_path := kwargs.get('embeddings')):
        existing_models['embeddings'] = pd.read_csv(embeddings_path)
    if (vectorizer_path := kwargs.get('vectorizer')):
        existing_models['vectorizer'] = load(vectorizer_path)

    # Predictive models
    if (linear_svc_path := kwargs.get('linear_svc')):
        existing_models['linear_svc'] = (load(linear_svc_path), {}) # Place holder for performance metrics
    if (multi_nb_path := kwargs.get('multi_nb')):
        existing_models['multi_nb'] = (load(multi_nb_path), {}) # Place holder for performance metrics

    return signals, (raw_tweets_df, existing_models)


# ## Build `transform` function



def transform(raw_tweets_df, sentiment_map, column_mappings={},
                                filter_words=[], existing_models={}, **kwargs):
    signals = []
    _, clean_datasets = cleanAndFilter(raw_tweets_df=raw_tweets_df,
                                                    column_mappings=column_mappings,
                                                    filter_words=filter_words,
                                                    signals=signals,
                                                    **kwargs)
    clean_tweets, filtered_tweets = clean_datasets

    # Gather/build word vector model
    if (existing_word_vec := existing_models.get('word_vec')):
        word_vectors = existing_word_vec
    else:
        _, word_vectors = buildWordVectors(clean_tweets, signals=signals, **kwargs)
        existing_models['word_vec'] = word_vectors

    # Gather/build clustering model
    if (existing_kmeans := existing_models.get('kmeans')):
        cluster_model = existing_kmeans
    else:
        # Build KMeans model to cluster words into positive, negative, and neutral clusters
        if (kmeans_params := kwargs.get('kmeans_params')):
            cluster_model = KMeans(**kmeans_params)
            existing_models['kmeans'] = cluster_model
        else:
            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_NECCESSARY,
                                            'Need parameters for the KMeans clustering algorithm.'))
            return signals, None

        # Train the cluster model
        if (kmeans_train := kwargs.get('kmeans_train')):
            cluster_model = cluster_model.fit(X=word_vectors.vectors.astype('double'), **kmeans_train)
        else:
            cluster_model = cluster_model.fit(X=word_vectors.vectors.astype('double'))

    ############# Get User Input for Sentiment Assignment ###############
    _, cluster_sentiment_defs = setClusterSentiment(vectors=word_vectors,
                                                    model=cluster_model,
                                                    sentiment_mapping=sentiment_map,
                                                    signals=signals,
                                                    display_terms=kwargs.get('display_terms'))
    if not cluster_sentiment_defs:
        return signals, None
    #####################################################################

    print('\nApplying sentiment mapping...')

    # Gather/generate word embeddings
    if (existing_embeddings := existing_models.get('embeddings')):
        words_df = existing_embeddings
    else:
        words_df = buildWordEmbeddings(word_vectors=word_vectors,
                                        model=cluster_model,
                                        sentiment_defs=cluster_sentiment_defs,
                                        sentiment_map=sentiment_map)
        existing_models['embeddings'] = words_df

    # Get the sentiment for the entire tweet
    if (threshold := kwargs.get('sentiment_threshold')):

        words_cluster_dict = dict(zip(words_df.words, words_df.cluster_value))

        def getSentiment(row):
            words_list = row['clean_tweet_words']
            total = sum(int(words_cluster_dict.get(word, 0)) for word in words_list)
            avg = total / len(words_list) if words_list else 0
            return -1 if (avg < -threshold) else 1 if (avg > threshold) else 0

        # Add sentiment column (integer values)
        filtered_tweets["sentiment_val"] = filtered_tweets.apply(getSentiment, axis=1)
        # Map integer sentiment to word value
        filtered_tweets["sentiment"] = filtered_tweets["sentiment_val"].map(sentiment_map)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_NECCESSARY,
                                        'Need sentiment threshold parameter to assign sentiment values.'))
        return signals, None

    # Confirm sentiment distribution with user
    if not peekSentimentDistrib(filtered_tweets):
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.USER_INPUT,
                                        'Distribution was unsatisfactory.'))
        return signals, None

    return signals, (filtered_tweets, words_df, cluster_sentiment_defs)


# ## Build `model` function



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
    test_size = 0.25 # default value provided by scikit-learn
    if (config_test_size := kwargs.get('test_size')):
        test_size = config_test_size
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                        'No test size provided for model training. Will use default.'))

    train_test_df, _ = train_test_split(agg_sentiment_df, test_size=test_size, random_state=10)

    X = train_test_df['clean_tweet']
    y = train_test_df['sentiment_val']

    # Split the dataset set into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Instantiate TfidfVectorizer
    if (existing_vectorizer := existing_models.get('vectorizer')):
        vectorizer = existing_vectorizer
    else:
        if (vectorizer_params := kwargs.get('vectorizer_params')):
            vectorizer = TfidfVectorizer(**vectorizer_params)
        else:
            signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                        'No provided argument for `TfidfVectorizer` object. Will use defaults.'))
            vectorizer = TfidfVectorizer(min_df=3, stop_words='english')

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
        if (linearSVC_params := kwargs.get('linear_svc_params')):
            linearSVC = LinearSVC(**linearSVC_params)
        else:
            signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                        'No provided argument for `LinearSVC` object. Will use defaults.'))
            linearSVC = LinearSVC()

        # Fit the model
        linearSVC.fit(X_train_tf, y_train)

    svc_performance = performanceSummary(model=linearSVC, X_test=X_test_tf, y_test=y_test)

    ## MultinomialNB ##

    if (existing_multi_nb := existing_models.get('multi_nb')):
        # (NOTE: Must use same vectorizer from its fitting)
        multiNB = existing_multi_nb
    else:
        # Instantiate the model
        if (multiNB_params := kwargs.get('multi_nb_params')):
            multiNB = MultinomialNB(**multiNB_params)
        else:
            signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS,
                                        'No provided argument for `MultinomialNB` object. Will use defaults.'))
            multiNB = MultinomialNB()

        # Fit the model
        multiNB.fit(X_train_tf, y_train)

    nb_performance = performanceSummary(model=multiNB, X_test=X_test_tf, y_test=y_test)

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
                                            if (name in built_models and built_models[name] is not None) }
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
                if line == '# ## Execute `pipeline`':
                    term_index = i
                    break
                elif '# In[' in line and not "SKIP" in line:
                    cell_markers.add(i)

            fp.writelines([l for i, l in enumerate(lines[:term_index]) if i not in cell_markers])
    return signals



# ## Build `pipeline` function



def pipeline1(default_configs, user_configs=None, extract_args={}, transform_args={}, model_args={}, load_args={}, log_level=None):
    # Parse arguments
    parsing_signals, valid_params = buildConfig(dflt_configs=default_configs, usr_configs=user_configs,
                                                    extract_config=extract_args, transform_config=transform_args,
                                                    model_config=model_args, load_config=load_args)
    processSignals(signals=parsing_signals, log_level=log_level) # Process error/info signals
    extract_params, transform_params, model_params, load_params = itemgetter(*['EXTRACT', 'TRANSFORM', 'MODEL', 'LOAD'])(valid_params)

    print('\n--- Executing Pipeline 1. ---\n')

    # Store run-specific information
    execution_config = defaultdict(dict)

    ## Extract (import)
    print('Stage 1: Extracting...')
    extract_signals, extracted_data = extract(**extract_params)
    processSignals(signals=extract_signals, generated_files=load_params, log_level=log_level) # Process error/info signals
    raw_tweets_df, imported_models = extracted_data
    print('Completed Stage 1.', end='\n\n')

    # print(default_configs)
    # print(raw_tweets_df.columns)

    ## Transform
    print('Stage 2: Transforming...')
    transform_signals, transform_data = transform(raw_tweets_df=raw_tweets_df, existing_models=imported_models,
                                                    **transform_params)
    processSignals(signals=transform_signals, generated_files=load_params, log_level=log_level) # Process error/info signals
    tweet_sentiment_df, word_vecs, sentiment_defs = transform_data

    # Store sentiment encodings
    execution_config['sentiment_vals'] = {
        'value_mapping': transform_params['sentiment_map'],
        'cluster_mapping': sentiment_defs
    }
    print('Completed Stage 2.', end='\n\n')

    ## Modeling
    if model_params.get('build_models'):
        print('Stage 2.5: Modeling...')
        model_signals, model_data = model(sentiment_df=tweet_sentiment_df, existing_models=imported_models,
                                            **model_params)
        processSignals(signals=model_signals, generated_files=load_params, log_level=log_level) # Process error/info signals

        imported_models.update(model_data) # Update previously imported models
        print('Completed Stage 2.5.', end='\n\n')

    ## Loading (export)
    print('Stage 3: Loading...')
    load_signals = load(transform_df=tweet_sentiment_df, built_models=imported_models,
                            destinations=load_params, config=execution_config)
    processSignals(signals=load_signals, generated_files=load_params, log_level=log_level) # Process error/info signals
    print('Completed Stage 3.', end='\n\n')
    print('<done>')

    return tweet_sentiment_df, word_vecs, imported_models

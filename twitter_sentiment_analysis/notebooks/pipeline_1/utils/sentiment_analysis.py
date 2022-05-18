
# Import dependencies
from shared_imports import (defaultdict, np, pd, 
                            ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS)

from IPython.display import display
from sklearn.preprocessing import MinMaxScaler

## Need user input to determine each cluster's sentiment ##

def setClusterSentiment(vectors, model, sentiment_mapping, signals=[], **kwargs):
    display_terms = kwargs.get('display_terms', 20)
    print(f'** Top {display_terms} Similar Word Vectors By Cluster **\n')

    collectSamples(vectors=vectors, model=model)

    ## Get input
    map_string = ', '.join([f'{k} = {v}' for k, v in sentiment_mapping.items()])
    print(f"\nLabel each cluster: {map_string} (\"r\" for new samples, \"q\" to exit)")
    cluster_sentiment_defs = []
    user_input = ''
    batch_number = 0
    while len(cluster_sentiment_defs) < len(sentiment_mapping)-1 and user_input != 'q':
        
        user_input = input(f'Cluster {len(cluster_sentiment_defs)} value:')
        if user_input == 'q':
            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.USER_INPUT))
            return signals, None

        if user_input == 'r':
            print(f'\n\nGenerating next {display_terms} samples...\n')
            batch_number += 1
            collectSamples(vectors=vectors, model=model, batch=batch_number)
            print('Current state:', cluster_sentiment_defs)
            print('Setting cluster:', len(cluster_sentiment_defs))
            continue

        try:
            value = int(user_input)
            if value in cluster_sentiment_defs:
                print('Already used this sentiment value.')
                continue
            elif value not in range(-1, 2):
                print(f'Value not in provided sentiment mapping: {list(sentiment_mapping.keys())}')
                continue

            cluster_sentiment_defs.append(value)
            print(f'Set cluster {len(cluster_sentiment_defs)-1} to {value} ({sentiment_mapping[value]})')
        except ValueError:
            print(f'Must input a number in range {list(sentiment_mapping.keys())}. Press q to exit')
            
    cluster_sentiment_defs.append((set(sentiment_mapping.keys()) - set(cluster_sentiment_defs)).pop())
    print((f'Set cluster {len(cluster_sentiment_defs)-1} to {cluster_sentiment_defs[-1]}' +
            f'({sentiment_mapping[cluster_sentiment_defs[-1]]})'))
    
    return signals, cluster_sentiment_defs


def buildWordEmbeddings(word_vectors, model, sentiment_defs, sentiment_map):
     # Create a DataFrame of words with their embeddings and cluster values
    words_df = pd.DataFrame(word_vectors.index_to_key, columns=['words'])
    words_df['vectors'] = words_df.words.apply(lambda x: word_vectors[str(x)])
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


def collectSamples(vectors, model, batch=0):
    num_clusters = model.cluster_centers_.shape[0]
    word_vec_list = [vectors.similar_by_vector(model.cluster_centers_[x], 
                                                    topn=(25 * (batch+1)), 
                                                    restrict_vocab=None) 
                                                        for x in range(num_clusters)]
    
    cluster_values = np.array(list(zip(*[x[(25 * batch):] for x in word_vec_list])))
    cluster_cols = [f'Cluster {x}' for x in range(num_clusters)]

    # # Collect terms spanning multiple clusters for deciphering
    term_freq, counts = np.unique([x[0] for x in np.vstack(cluster_values)], axis=0, return_counts=True)
    unique_terms = term_freq[counts == 1]

    # Separate unique from duplicate terms
    unique_cluster_vals = [[] for _ in range(num_clusters)]
    shared_cluster_vals = defaultdict(lambda : [0] * num_clusters)

    for ix, iy in np.ndindex(cluster_values.shape[:2]):
        term, vec = cluster_values[ix, iy]
        if term in unique_terms:
            unique_cluster_vals[iy].append((term, float(vec)))
        else:
            shared_cluster_vals[term][iy] = float(vec)


    print('Unique Terms from Clusters')
    max_num_unique = max(len(c) for c in unique_cluster_vals)

    # Sort by and drop vector. Even out column lengths
    unique_cluster_terms = np.array([[val[0] for val in sorted(cluster, key=lambda x: x[1])] + 
                                        ['']*(max_num_unique-len(cluster)) # Adjust lengths
                                            for cluster in unique_cluster_vals])

    unique_terms_df = pd.DataFrame(unique_cluster_terms.T, columns=cluster_cols)
    display(unique_terms_df)

    print('\nDuplicate Terms from Clusters')
    if shared_cluster_vals:
        # Build dict for scaling
        shared_vals_df = pd.DataFrame.from_dict(shared_cluster_vals, orient='index', 
                                                    columns=cluster_cols).reset_index()

        # Calc differences between clusters (for interpretation purposes)
        for c in cluster_cols[1:]:
            shared_vals_df[f'{c} relative to {cluster_cols[0]}'] = shared_vals_df[cluster_cols[0]] - shared_vals_df[c]

        shared_vals_df = shared_vals_df.drop(cluster_cols[1:], axis=1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_cols = [c for c in shared_vals_df.columns if c not in ['index', cluster_cols[0]]]

        scaled_data = scaler.fit_transform(shared_vals_df[scaled_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_cols)
        duplicate_terms_df = pd.merge(shared_vals_df[['index', cluster_cols[0]]], 
                                        scaled_df, left_index=True, right_index=True) \
                                            .sort_values(by=cluster_cols[0])

        col_map = { cluster_cols[0]: f'{cluster_cols[0]} (baseline)', 'index': 'term'}
        duplicate_terms_df = duplicate_terms_df.rename(columns=col_map).set_index('term')
        display(duplicate_terms_df)
    else:
        print('\tNo duplicates between clusters')


def peekSentimentDistrib(tweets_df):
    print('\nCalculated Sentiment Distribution:')
    display(tweets_df['sentiment'].value_counts())
    user_input = input('Distribution okay? (y/n) ')
    if user_input != 'y':
        return False
    return True


def Opt1_cluster(word_vectors, tweets_df=None):

    # Scale data.
    X_scaled = preprocessing.scale(ord_vectors.vectors.astype('double'))

    # Standardize features
    X_std = StandardScaler().fit_transform(X_scaled)

     # Using PCA to remove columns (features) which have less co-relation
    n_clusters = 3
    word_vectors_pca = PCA(n_components=n_clusters)
    word_vectors_pca = word_vectors_pca.fit(X_std)
    pca_matrix = word_vectors_pca.transform(X_std)

    # Initialize dataframe that will hold all calculated output during clustering
    words_df = pd.DataFrame({
                            'words': word_vectors.index_to_key,
                            'vectors': list(X_std),
                            'pca_values': list(pca_matrix)
                        })

    # Define clustering model (KMeans algo)
    cluster_model = KMeans(n_clusters=n_clusters, max_iter=400)
    cluster_model = cluster_model.fit(pca_values)

    # Generate cluster predictions for each sample (word)
    X_ = np.stack(words_df.pca_values.to_numpy().ravel()).reshape(-1, n_clusters)
    words_df['predicted_cluster'] = cluster_model.predict(X_)

    # Use the model to translate each sample's features into "model-space" or "cluster-space"
    to_cluster_space = lambda x: cluster_model.transform([x]).flatten()
    words_df['cluster_space'] = words_df.pca_values.apply(to_cluster_space)


    # Group data by its predicted cluster and collect each sample's pca features and their "cluster-space" coordinantes
    # clusters_distances = np.flip(words_df.groupby(by='predicted_cluster').cluster_space.apply(
    #                 lambda x: np.stack(x.to_numpy())).to_numpy())
    # clusters_pca = np.flip(words_df.groupby(by='predicted_cluster').pca_values.apply(
    #                     lambda x: np.stack(x.to_numpy())).to_numpy())

    def func(df):
        c_stack = np.stack(df['cluster_space'].to_numpy().ravel())
        p_stack = np.stack(df['pca_values'].to_numpy().ravel())
        return pd.Series(data=[c_stack, p_stack])


    group_pairs = subset.groupby(by='predicted_cluster')[['cluster_space', 'pca_values']] \
                                            .apply(lambda x: func(x)).to_numpy().T


    def score(row):

        # ### sample transformed - all samples' transformed PCA in cluster
        # pca_query = row.pca_values.reshape(1, -1)

        # calculated_pca_diffs = np.array([euclidean_dist(clusters_pca[i], pca_query)
        #                                             for i in range(len(clusters_pca))], dtype=object) # distance from this sample to all others

        # avg_calculated_pca_diff = np.fromiter((np.mean(d[d != 0]) for d in calculated_pca_diffs), float) # between this sample and all other clusters


        # ### sample transformed - all samples' transformed PCA in cluster
        # transformed_query = row.cluster_space.reshape(1, -1)

        # calculated_transformed_diffs = np.array([euclidean_dist(clusters_distances[i], transformed_query)
        #                                             for i in range(len(clusters_distances))], dtype=object) # distance from this sample to all others

        # avg_calculated_transformed_diff = np.fromiter((np.mean(d[d != 0]) for d in calculated_transformed_diffs), float) # between this sample and all other clusters

        calc_distances = lambda q: np.array([euclidean_dist(group_pairs[i], [q]) for i in range(len(group_pairs))], dtype=object)
        cluster_dists = calc_distances(distance_query).reshape(2, 3)
        pca_dists = calc_distances(pca_query).reshape(2, 3)

        calc_avg_distances = lambda dists: np.fromiter((np.mean(d[d != 0]) for i in range(len(dists)) for d in dists[i]), float)
        avg_cluster_dists = calc_avg_distances(cluster_dists)
        avg_pca_diff = calc_avg_distances(pca_dists)



        ### sample transformed - cluster centers
        predicted_pca_diff = euclidean_dist(cluster_model.cluster_centers_, [pca_query, distance_query]).flatten()


        ## Define weights
        transformed_weight = avg_calculated_transformed_diff * safe_std_inverse(calculated_transformed_diffs)
        pca_weight = avg_calculated_pca_diff * safe_std_inverse(calculated_pca_diffs)

        ## Use models prediction distance as scalar "normalizer"
        predict_dist_std = np.std(clusters_distances[row.predicted_cluster])
        prediction_weights = row.cluster_space * ((1 / predict_dist_std) if predict_dist_std != 0 else 0)

        ## Define errors
        transformed_error = row.cluster_space - avg_calculated_transformed_diff
        pca_errror = row.cluster_space - avg_calculated_pca_diff

        prediction_pce_error = (predicted_pca_diff / (pca_errror * pca_weight))
        prediction_transformed_error = (predicted_distance_diff / (transformed_error * transformed_weight))


        prediction_error = np.mean([prediction_transformed_error, prediction_pce_error])

        prediction_confidence = (1 / (prediction_error))  * prediction_weights #* prediction_weights

        cluster_similarity = 1 / avg_calculated_transformed_diff

        row['confidence_scores'] = prediction_confidence
        row['cluster_similarity'] = cluster_similarity

        return row


    scored_words = words_df.apply(score, axis=1)

    # scoring_dimensions = ['cluster_space', 'cluster_scores', 'confidence_score']
    # words_clusters_df = scored_clusters[['words', 'predicted_cluster']+scoring_dimensions]
    # words_clusters_df = words_clusters_df.set_index('words')

    # cluster_sentiment_defs = setClusterSentiment(words_df, fit_kmeans, sentiment_map, version=1)


    return scored_words, cluster_model


opt1_cluster_df, opt1_model = Opt1_cluster(WORD_VECS)
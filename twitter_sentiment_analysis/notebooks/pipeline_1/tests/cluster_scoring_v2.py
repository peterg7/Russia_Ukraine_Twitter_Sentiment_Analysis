
from itertools import combinations

INCLDE_SELF_COMPARISON = False
n_comparison_cols = n_clusters if INCLDE_SELF_COMPARISON else n_clusters-1

## HELPERS
# z = nd_EuclideanDistance(cluster_space_matrix, x)
#         return np.repeat(z.T[np.newaxis, :, :], 3, 0)

def calcDistance(mat_A, mat_B):

    def applyEuclidean(subset, axis):
        distance = euclidean3d(mat_A, subset)
        adj_distance = distance.T[np.newaxis, :, :]
        return np.repeat(adj_distance, 3, 0)

    return np.apply_over_axes(applyEuclidean, mat_B, (0,1))


def euclidean3d(mat_A, mat_B):
    subs = mat_A - mat_B
    return np.sqrt(np.einsum('ijk,ijk->ij', subs, subs))

def generate_transform_weights(cluster_distances):
    return np.reciprocal(np.std(cluster_distances, axis=1))

def generateClusterQueries(cluster_matrix):
    cluster_queries = []
    n_cluster = len(cluster_matrix)
    for i in range(len(cluster_matrix)):
        raw_values = cluster_matrix[i][flat_mask[i]]
        query = np.repeat(raw_values[np.newaxis, :, :], n_comparison_cols, axis=0)
        cluster_queries.append(query)
    return cluster_queries

def explode(s, n_clusters):
    exploded_arr = pd.DataFrame(s.tolist(), s.index).to_numpy()
    z = np.zeros((largest_cluster_size, n_clusters))
    z[0:len(exploded_arr)] = exploded_arr
    return z


# lambda s: np.pad(s, pad_width=(largest_cluster_size-len(s), 0), mode='constant', constant_values=np.nan).shape

# Useful properties
cluster_sizes = np.bincount(opt1_cluster_df.predicted_cluster)
n_clusters = len(cluster_sizes)
largest_cluster_size = max(cluster_sizes)
flat_mask = np.ma.fromfunction(lambda x, y: y < cluster_sizes[x], (n_clusters, largest_cluster_size), dtype=int)

cluster_pair_ranges = [
    (range(n_clusters), n_comparison_cols),
    (range(n_comparison_cols, (n_comparison_cols-n_clusters), -1), n_comparison_cols)
]
cluster_pairs = sorted((combinations(*cluster_pair_ranges[0])), reverse=True)
inv_cluster_pairs = sorted((combinations(*cluster_pair_ranges[1])), reverse=True)


# Build space matrix
grouped_cluster_spaces = opt1_cluster_df.groupby(by='predicted_cluster').cluster_space.apply(lambda x: explode(x))
cluster_space_matrix = np.stack(grouped_cluster_spaces.to_numpy().ravel())


## ONLY COMPUTING DISTANCE WITH OTHER TWO CLUSTERS


# Generate queries
cluster_queries = generateClusterQueries(cluster_space_matrix)

# Calculate distance between each cluster
cluster_distances = [nd_EuclideanDistance(cluster_space_matrix[cluster_pairs[i],0:cluster_sizes[i]],
                                           cluster_queries[i]) for i in range(n_clusters)]

# Get the average of each cluster's distances
avg_cluster_dists = np.array([np.mean(dist, axis=1) for dist in cluster_distances])

# Find the standard deviation for each cluster
cluster_distance_stds = [np.std(cluster_distances[i]) for i in range(n_clusters)]


# Format the center of each cluster
queryable_centers = np.repeat(opt1_model.cluster_centers_[:, np.newaxis, :], n_comparison_cols, axis=1)


####
transformed_weight = np.dot(avg_cluster_dists, generate_transform_weight(cluster_dists))
####


# Get distance from each cluster to its center
predicted_distance_diffs = [nd_EuclideanDistance(queryable_centers[inv_cluster_pairs[i]],
                                                  cluster_queries[i]) for i in range(n_clusters)]

# Generate weights from the cluster distances
prediction_weights = cluster_space_matrix * np.reciprocal(cluster_distance_stds)


# Define the transformation error
transformed_error = row.cluster_space - avg_cluster_dists












########################## OFFICIAL USE #########################
#                                                               #


# Scale data.
X_scaled = preprocessing.scale(word_vectors.vectors.astype('double'))

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
cluster_model = cluster_model.fit(pca_matrix)

# Generate cluster predictions for each sample (word)
X_ = np.stack(words_df.pca_values.to_numpy().ravel()).reshape(-1, n_clusters)
words_df['predicted_cluster'] = cluster_model.predict(X_)

# Use the model to translate each sample's features into "model-space" or "cluster-space"
to_cluster_space = lambda x: cluster_model.transform([x]).flatten()
words_df['cluster_space'] = words_df.pca_values.apply(to_cluster_space)





#################################################################
#                                                               #
# ********************* Similarity Scoring ******************** #


'''
INPUT
    ==> word_vectors [gensim.Word2Vec]. Contails embeddings + full corpus
    ==> tweets_df [pandas.DataFrame]. Holds the information which formed the word vectors


OUTPUT
    ==> words_df [pandas.DataFrame]. Holds cluster (and sentiment) predictions
'''
from itertools import combinations

## Helper Functions ##


def nd_EuclideanDistance(word_vectors, query_vectors):
    subs = word_vectors - query_vectors
    return np.sqrt(np.einsum('ijk,ijk->ij', subs, subs))


def explodeAndPad(series_group, n_clusters):
    exploded_arr = pd.DataFrame(series_group.tolist(), series_group.index).to_numpy()
    z = np.zeros((max_cluster_size, n_clusters))
    z[0:len(exploded_arr)] = exploded_arr
    return z


def scalePredictionDistance(arr):
    std = np.std(arr, axis=1)
    return np.divide(1, std, out=np.zeros_like(std), where=std!=0)


def generateClusterQueries(cluster_matrix):
    cluster_queries = []
    for i in range(len(cluster_matrix)):
        raw_values = cluster_matrix[i]
        query = np.repeat(raw_values[np.newaxis, :, :], n_clusters, axis=0)
        cluster_queries.append(query)
    return cluster_queries




## Constants

cluster_sizes = np.bincount(opt1_cluster_df.predicted_cluster)
max_cluster_size = max(cluster_sizes)
n_clusters = len(cluster_sizes)

# Boolean mask used for filtering and reshaping
cluster_space_mask = np.ma.fromfunction(lambda x, y: y < cluster_sizes[x],
                                                (n_clusters, max_cluster_size),
                                                    dtype=int)

cluster_space_centers = cluster_model.cluster_centers_[:, np.newaxis, :]


##### OVERALL ####
## Check dimensions!!!
##      Includes the random Transposes
##      What are the final dimensions????????
## Get a hold on all of the np.array declarations


# Build a transformation matrix with the cluster-space data
grouped_cluster_spaces = opt1_cluster_df.groupby(by='predicted_cluster').cluster_space \
                                            .apply(lambda x: explodeAndPad(x))
cluster_space_matrix = np.stack(grouped_cluster_spaces.to_numpy().ravel())



# Structure query data. Contains every sample separated by its predicted cluster value
cluster_queries = generateClusterQueries(cluster_space_matrix)

# Compute the euclidean distance between all clusters and those in each predicted cluster group
# Option 1)
cluster_distances = np.array([ nd_EuclideanDistance(cluster_space_matrix, cluster_queries[i]).T ## DON'T LIKE THIS TRANSPOSE!!!
                                    for i in range(n_clusters) ])

# Options 2)
diff = nd_EuclideanDistance(cluster_space_matrix, cluster_space_matrix.T)
diff.shape

# Average the distances acquired above
avg_cluster_dists = np.array([np.mean(dist, axis=0) for dist in cluster_distances])

# Determine the std for each cluster
cluster_distance_stds = np.array([np.std(cluster_distances[i]) for i in range(n_clusters)])

# Get the distance of each sample to each cluster's center (a very stong indication of similarity)
predicted_distance_diffs = np.array([nd_EuclideanDistance(queryable_centers,
                                                    cluster_queries[i]).T for i in range(n_clusters)])

# Generate weights for the confidence level based on the distance between clusters
transformed_weight = np.dot(avg_cluster_dists,
                                scalePredictionDistance(cluster_distances))[:, np.newaxis, :]

# Build weights based on the spread of the clusters and how "wide" of an area they cover
prediction_weights = cluster_space_matrix * np.reciprocal(cluster_distance_stds)

# Calculate the model's average error using the distance of each sample and each cluster's average distance
transformed_error = np.array([nd_EuclideanDistance(cluster_space_matrix,
                                                    avg_cluster_dists[:, np.newaxis, :]).T
                                for i in range(n_clusters)])

# Build an error score for the model's transformation
prediction_error = np.divide(predicted_distance_diffs, (transformed_error * transformed_weight))

# Finally, combine the overall predictions' error and weight to build a confidence score
prediction_confidence = np.reciprocal((prediction_error))  * prediction_weights

# Store the relative similarities between each cluster
global_cluster_similarity = np.divide(1, avg_cluster_dists, out=np.zeros_like(avg_cluster_dists),
                                    where=avg_cluster_dists!=0)

# Also maintain a similarity score with each cluster for every sample
cluster_similarity = np.divide(1, cluster_distances, out=np.zeros_like(cluster_distances),
                                    where=cluster_distances!=0)







## Just some cool stuff

def sum_axis_i(arr, axis, i):
    idx = (np.s_[:],) * axis + (i,)
    print(idx)
    return arr[idx]

sum_axis_i(q, 0, 0)


idx = (np.s_[:],0,1)
print(idx)
cluster_space_matrix[idx]

cluster_space_matrix[(0, np.s_[:],np.s_[:])]






























#### Helper Functions ####

def explodeAndPad(series_group, cluster_dims):
    exploded_arr = pd.DataFrame(series_group.tolist(), series_group.index).to_numpy()
    z = np.zeros((max(cluster_dims), len(cluster_dims)))
    z[0:len(exploded_arr)] = exploded_arr
    return z


def scalePredictionDistance(arr):
    std = np.std(arr, axis=1)
    return np.divide(1, std, out=np.zeros_like(std), where=std!=0)


def generateClusterQueries(cluster_matrix):
    cluster_queries = []
    for i in range(len(cluster_matrix)):
        raw_values = cluster_matrix[i]
        # query = np.repeat(raw_values[np.newaxis, :, :], n_clusters, axis=0).T
        query = raw_values[np.newaxis, :, :] #.T ??
        cluster_queries.append(query)
    return np.squeeze(np.array(cluster_queries), axis=1)


def calcDistance(mat_A, mat_B):

    def applyEuclidean(subset, axis):
        distance = euclidean3d(mat_A, subset)
        adj_distance = distance.T[np.newaxis, :, :]
        return np.repeat(adj_distance, 3, 0)

    return np.apply_over_axes(applyEuclidean, mat_B, (0,1))


def euclidean3d(mat_A, mat_B):
    subs = mat_A - mat_B
    return np.sqrt(np.einsum('ijk,ijk->ij', subs, subs))


matrixDot2d = lambda mat_A, mat_B: np.einsum('ij,jk->ik', mat_A, mat_B)
matrixDot3d = lambda mat_A, mat_B: np.einsum('ijk,ijl->ikl',mat_A, mat_B)
matrixMult2d = lambda mat_A, mat_B: np.einsum('ij, jk -> ik', mat_A, mat_B)
matrixMult3d = lambda mat_A, mat_B: np.einsum('nmk,nkj->nmj', mat_A, mat_B)
matrixMult3d_2d = lambda mat_A, mat_B: np.einsum('ijk,ik->ijk', mat_A, mat_B)


##### Scoring Process #####

# Define independent constants

# Setup a matrix for comparisons to the coordinates of each cluster's center
cluster_space_centers = cluster_model.cluster_centers_[:, np.newaxis, :]

# Define specs about the clustering process
cluster_sizes = np.bincount(word_predictions.predicted_cluster)
max_cluster_size = max(cluster_sizes)
n_clusters = len(cluster_sizes)

# Boolean mask used for filtering and reshaping
masking_func = lambda x, y: y < cluster_sizes[x]
cluster_space_mask = np.ma.fromfunction(masking_func,
                                            (n_clusters, max_cluster_size), dtype=int)

# Build a transformation matrix with the cluster-space data (count X samples X value)
grouped_cluster_spaces = word_predictions.groupby(by='predicted_cluster').cluster_space \
                                    .apply(lambda x: explodeAndPad(x, cluster_sizes))
cluster_space_matrix = np.stack(grouped_cluster_spaces.to_numpy().ravel())

# Structure query data. Contains every sample separated by its predicted cluster value
cluster_queries = generateClusterQueries(cluster_space_matrix)

# Compute the euclidean distance between all clusters and those in each predicted cluster group
cluster_distances = calcDistance(cluster_space_matrix, cluster_queries)

# Average the distances acquired above
avg_cluster_dists = np.mean(cluster_distances, axis=1)

# Determine the std for each cluster
cluster_distance_stds = np.std(cluster_distances, axis=1)

# Get the distance of each sample to each cluster's center (a very stong indication of similarity)
predicted_distance_diffs = calcDistance(cluster_space_centers, cluster_queries)

# Generate weights for the confidence level based on the distance between clusters
transformed_weights = np.dot(avg_cluster_dists,
                                scalePredictionDistance(cluster_distances)) # [:, np.newaxis, :] ??

# Build weights based on the spread of the clusters and how "wide" of an area they cover
prediction_weights = np.moveaxis(cluster_space_matrix, 1, 0) * np.reciprocal(cluster_distance_stds[np.newaxis, :, :])

# Calculate the model's average error using the distance of each sample and each cluster's average distance
transformed_distances = euclidean3d(cluster_space_matrix, avg_cluster_dists[:, np.newaxis, :])
transformed_error = matrixDot2d(cluster_distance_stds, transformed_distances)

# Set up matricies in preparation for
inverse_predicted_distance = np.linalg.pinv(predicted_distance_diffs)
weighted_transform_err = matrixMult2d(transformed_weights, transformed_error)

# Build an error score for the model's transformation
prediction_error = matrixMult3d_2d(inverse_predicted_distance, weighted_transform_err)

# Finally, combine the overall predictions' error and weight to build a confidence score
reciprocal_prediction_err = np.reciprocal(prediction_error)
prediction_confidence = matrixDot3d(reciprocal_prediction_err.T, prediction_weights)

# Store the relative similarities between each cluster
global_cluster_similarity = matrixMult2d(np.linalg.pinv(avg_cluster_dists), weighted_transform_err)

# Also maintain a similarity score with each cluster for every sample
inverse_cluster_distance = np.moveaxis(np.linalg.pinv(cluster_distances), 2, 1)
inverse_transform_distances = np.linalg.pinv(transformed_distances)

sample_similarity = matrixMult3d_2d(inverse_cluster_distance, cluster_distance_stds)
sample_similarity += matrixMult2d(cluster_distance_stds, inverse_transform_distances.T)[...,np.newaxis]

result_scores = {
    'confidence_score': prediction_confidence,
    'global_similarity': global_cluster_similarity,
    'sample_similarity': sample_similarity
}
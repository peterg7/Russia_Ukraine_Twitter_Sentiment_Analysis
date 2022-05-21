
### Define constants/independent variables

# Cluster data specs/sizes
cluster_sample_sizes = np.bincount(word_predictions.predicted_cluster)
n_clusters, n_features = cluster_model.cluster_centers_.shape
n_samples = sum(cluster_sample_sizes)
smallest_cluster, largest_cluster = min(cluster_sample_sizes), max(cluster_sample_sizes)
cluster_space_dims = (n_clusters, largest_cluster, n_features)

# Setup a matrix for comparisons to the coordinates of each cluster's center
cluster_space_centers = cluster_model.cluster_centers_[:, np.newaxis, :]

# Boolean mask used for filtering and reshaping
masking_func = lambda x, y: y < cluster_sample_sizes[x]
cluster_space_mask = np.ma.fromfunction(masking_func, (n_clusters, largest_cluster), dtype=int)


# Utility function to undo the transformation necessary for scaling 3d matricies
restoreMat3d = lambda mat: mat.reshape(*cluster_space_dims)



### Build a transformation matrix with the cluster-space data


# Group the data by the predicited cluster value
cluster_groups = word_predictions.groupby(by='predicted_cluster').cluster_space \
                                    .apply(lambda x: explodeAndPad(x, cluster_sample_sizes))


# Convert the grouped data into a usable numpy array
cluster_space = np.stack(cluster_groups.to_numpy().ravel())


# Structure query data. Contains every sample separated by its predicted cluster value
cluster_queries = generateClusterQueries(cluster_space_matrix)


# Compute the euclidean distance between all samples and the entire space
sample_space = calcDistance(cluster_queries, cluster_space_matrix)


                ########## Basic Stats ##########


### Averages ###

# Average the distances acquired above to produce approximate cluster centers
SS_avg_internal_distances = np.mean(sample_space, axis=1)

# Using the modeled cluster space, find the average "distances" per cluster (all will be [3,3])
CS_avg_internal_distances = np.mean(cluster_space, axis=1)

'''
**** Notes ****
The averages of internal distances within sample space can be treated as
approximate cluster centers. i.e., right below this comment block
'''
sample_space_centers = SS_avg_internal_distances


### Standard Deviation ###

# Determine the std for each cluster
SS_internal_distances_std = np.std(sample_space, axis=1)

# Compute the std for each cluster using the cluster-space
CS_internal_distances_std = np.std(cluster_space, axis=1)


### Variance ###

# Determine the var of each cluster
SS_internal_distances_ar = np.var(sample_space, axis=1)

# Determine the var of each cluster
CS_internal_distances_ar = np.var(cluster_space, axis=1)


######## Differencing ########


# Get the distance of each sample to each cluster's center (a very stong indication of similarity)
CS_all_distances_all_centers = calcDistance(cluster_queries, cluster_space_centers)

# Get the distance of every sample from the average of their combined distances
SS_all_distances_all_centers = calcDistance(cluster_queries, sample_space_centers)

# Calculate the difference in average distances between the observed/calculated version and the one provied
# by the clustering model
avg_distances_of_center_distances = calcDistance(avg_model_distances, avg_observed_distances)
# or avg_center_distances_distancs


####### Inertia and distributions #######

# Invert the sample distance's to each cluster
center_similarity = np.linalg.pinv(model_center_diffs)


# Utilize KMeans' intertia property
samples_distribution = [(x/n_samples) for x in cluster_sample_sizes]

cluster_distance_metric = [x*cluster_model.inertia_ for x in samples_distribution]


# Build an inertia value by tracing KMeans' implementation
samples_df = np.stack(word_predictions.pca_values.to_numpy().ravel())
shortest_squared_distances = np.square(np.apply_along_axis(min, axis=1, arr=samples_df))
observed_inertia = np.sum(shortest_squared_distances)

observed_distance_metric = [x*observed_inertia for x in samples_distribution]
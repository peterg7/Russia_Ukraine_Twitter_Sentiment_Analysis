## Constants/Predefined

- cluster_sample_sizes
    - *list of length `n_clusters` and
each index has the number of samples stored
in that cluster*
- n_clusters
- n_features
- n_samples
- smallest_cluster
- largest_cluster
- cluster_space_dims
    - *tuple holding the three dimensions
    of the cluster space*

- cluster_space_centers
    - *a 3x3 array with the values being the
    center of each cluster in "cluster space"*


## Building "cluster space"
1. Group words by their predicted cluster value (provided by the kMeans model)

2. Convert to a 3D matrix of the form: (I x J x K) --> defines **cluster space**

3. Collect all of the sample data from the words dataset and again, group by
the predicted cluster value. It differs from the above array because it is
somewhat like a sparse matrix. The column lengths are filled to be of equal length
but the data does not do the same. Therefore, all but one column will be zero-padded.
This allows or quick (and typically effortless) analysis at the cluster level. This
form of the dataset is referred toas `cluster_queries` because of its natural ablility
to isolate clusters.

4. Next is creating the **"observable"** or **"sample"** space. This is done by calculating
the distance between the `cluster_queries` and `cluster_space`. This computation assess
every pairing of query to cluster space and doesn't rely on any other information from
the cluster space matrix or the clustering model.

5. Then to implement analytical techniques. The main purpose for this seup so far
has been to generate a useful, accurate, and efficient algorithm for determining
a "confidence score" for every sample. The score will be based on the similarity
and differences between the `cluster_space` and the `sample_space`. The easiest way to
start is with the fundamental statistical techniques such as mean, standard deviation,
variance, etc... These may come in handy later and are easy enough to implement that
there's not much downside. I've listed the ones I implemented below and included
the parameters I used as well as divided them into three groups. *Observable/Sample*,
*Cluster/Model*, and *Bridged*.

**Sample_Space**
- sample_space_avg
    - calculates the mean of the entire`sample_space` at the "sample level" (ie. on axis=1)
- sample_space_std
    - generates the standard deviation of the entire `sample_space`
- sample_space_var
    - outputs the varianceof the entire `sample_space`


**Cluster_Space**



**Bridged_Space**
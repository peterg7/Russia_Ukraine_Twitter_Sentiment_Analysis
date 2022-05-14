# Detailed outline of configurations

*If the provided confguration is not interpretable, the default config will be used entirely or to sumplement invalid entries*

## Basic format
A JSON or Python `dict` object with the following outer structure:
```json
{
  "EXTRACT": {},
  "TRANSFORM": {},
  "MODEL": {},
  "LOAD": {}
}
```

If any of the above entries are missing, they will be substituted with default values.

## Breakdown by stage

When items are required, default values have been included. All fields taged with `REQUIRED INPUT` will cause the pipeline to fail if missing or uninterpretable.

### EXTRACT

```python
"EXTRACT": {
  "data_import_path": "",				# Path (or url) to the raw data (REQUIRED INPUT)
  "data_import_version": None, 	# If there is a version associated with the file path
  "data_import_dest": "", 			# Where the raw data will be imported (REQUIRED INPUT)
  "new_dataset": False, 				# Will only pull data from source if True
  "word_vec": "", 							# Path to existing word2vec model
  "kmeans": "", 								# Path to existing kMeans model
  "embeddings": "", 						# Path to existing vector embeddings
  "vectorizer": "", 						# Path to existing vectorizer model
  "linear_svc": "",							# Path to existing LinearSVC model
  "multi_nb": ""								# Path to existing MultinomialNB model
}
```

### TRANSFORM

```python
"TRANSFORM": {
  "clean_tweet": [],  					# List of regex rules for cleaning each tweet
  "clean_hashtag": [], 					# List of regex rules for cleaning each hastag
  "column_mappings": { 					# Map of raw columns to expected columns 
    "": "username",
    "": "tweet",
    "": "hashtags"
  },
  "filter_words": [], 					# Only process tweets containing these words
  "word_vec_args": { 						# Used to build a word2vec model if no model extracted 
    # (see genism.models.Word2Vec)
    "vector_size": 300,
    "window": 5,
    "min_count": 4,
    "workers": multiprocessing.cpu_count()-1,
    "negative": 20,
    "sample": 1e-5,
    "alpha": 0.03,
    "min_alpha": 0.007,
    "seed": 42,
    "progress_per": 50000,
    "epochs": 30,
    "report_delay": 1
  }, 
  "kmeans_args": { 							# Used to build a kMeans model if no model extracted
    # (see sklearn.cluster.KMeans)
    "n_clusters": 3,
    "max_iter": 1000,
    "n_init": 50,
    "random_state": 42
  },
  "sentiment_threshold": 0.15, 	# Similarity threshold for distinguishing between clusters
  "sentiment_map": {  					# Define relationship between sentiment & sentiment_val
    -1: "negative",
    0: "neutral",
    1: "positive"
  },
  "display_terms": 25 					# Number of terms to show when assigning cluster sentiment
}
```

### MODEL

```python
"MODEL": {
  "build_models": True, 				# Control whether the pipeline should build models
  "vectorizer_args": { 					# Used to build a vectorizer model if no model extracted
    # (see sklearn.feature_extraction.text.TfidfVectorizer)
    "min_df": 3,
    "sublinear_tf": True,
    "ngram_range": (1,2),
    "stop_words": "english"
  }, 
  "linear_svc_args": { 								# For a LinearSVC model if no model extracted
    "random_state": 0
  },
  "multi_nb_args": { } 								# For a MultinomialNB model if no model extracted
}
```

### Load

```python
"LOAD": {
  "config": "", 								# Location to output the settings used for the execution
  "notebook": "", 							# Location to export this entire notebook as Python script
  "transform": "", 							# Location to store the transformed sentiment dataset
  "word_vec": "", 							# Path to save the word2vec model
  "kmeans": "", 								# Path to save the kMeans model
  "embeddings": "",							# Path to save the generated embeddings
  "vectorizer": "", 						# Path to save the vectorizer model
  "linear_svc": "",							# Path to save the LinearSVC model
  "multi_nb": ""								# Path to save the MultinomialNB model
}
```

### metadata

```python
"metadata": {
  "required_params": [
    "data_import_path",
    "data_import_dest",
    "column_mappings"
  ],
  "necessary_data_mappings": [
    "username", 
    "tweet", 
    "hashtags"
  ],
  "necessary_sentiment_values": [
    "negative",
    "neutral",
    "positive"
  ],
  "locations": [
    "data_import_path",
    "data_import_dest",
    "word_vec",
    "kmeans",
    "embeddings",
    "vectorizer",
    "linear_svc",
    "multi_nb"
  ]
}
```


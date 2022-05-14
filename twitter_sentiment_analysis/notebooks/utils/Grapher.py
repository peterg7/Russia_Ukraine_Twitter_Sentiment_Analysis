
# Import dependencies
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

POS_SENTIMENT_COLOR = '#f7e91b' # yellow
NEU_SENTIMENT_COLOR = '#1b97f7' # blue
NEG_SENTIMENT_COLOR = '#f71b29' # red
    
# Display a word cloud with the given text
def generateWordcloud(text):
    words=' '.join([words for words in text])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


# Make a pie chart from a dataframe's column distribution
def graphDistribution(df, plot_col, title='', fig_size=(7, 7)):
    fig = plt.gcf()
    fig.set_size_inches(fig_size)

    pie_df = df[plot_col].value_counts().reset_index()

    plt.pie(pie_df[plot_col],
            labels=pie_df["index"],
            radius=2,
            colors=[POS_SENTIMENT_COLOR, NEU_SENTIMENT_COLOR, NEG_SENTIMENT_COLOR],
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


# Shorthand functions for commonly used plots

graphWordDist = lambda word_vecs: graphDistribution(word_vecs, 'sentiment', 'Sentiment Distribution of Words')
graphTweetDist = lambda tweet_df: graphDistribution(tweet_df, 'sentiment', 'Sentiment Distribution of Tweets')

graphUsernames = lambda tweets_df: graphCounts(tweets_df, 'username', 'sentiment', 'Top 10 Highest Tweeting usernames', tweets_df['username'].value_counts().iloc[:10].index)
graphHashtags = lambda tweets_df: graphCounts(tweets_df, 'hashtags', 'sentiment', 'Top 10 Hashtags', tweets_df['hashtags'].value_counts().iloc[1:10].index, (15,10))



## Samples ##
'''
graphWordDistribution(word_vecs)
graphTweetDistribution(sentiment_df)

graphKeywordCounts(sentiment_df, 'clean_tweet', 'sentiment', ['russia'])
graphTopUsernames(sentiment_df)
graphTopHashtags(sentiment_df)

generateWordcloud(sentiment_df[sentiment_df['sentiment_val']==1]['clean_tweet'].values)
generateWordcloud(sentiment_df[sentiment_df['sentiment_val']==-1]['clean_tweet'].values)

graphConfusionmatrix(model_dict['LinearSVC']['conf_mat'], model_dict['sentiment_id'])
graphConfusionmatrix(model_dict['MultinomialNB']['conf_mat'], model_dict['sentiment_id'])

graphCoefficients(model_dict['LinearSVC']['model'], model_dict['features'])
'''
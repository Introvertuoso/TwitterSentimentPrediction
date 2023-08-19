# Project: Twitter Sentiment Prediction

This is our submission for the final graded project for the SS23 course *"Machine Learning"* at Saarland University.

In this project, we work with a dataset of tweets that have been labeled with both a sentiment score and
a sentiment type. We aim to predict these targets for new, unseen tweets.

We will be tackling two supervised tasks, namely, regression and classification over two targets. We hope
to predict a compound score reflecting a tweet’s sentiment for the former. At the same time, we would like to
predict the sentiment label in the latter.

The sentiment score is a continuous value between -1 and 1 that represents the degree of positivity or
negativity of the tweet. The sentiment type is a categorical value that can be one of three options: positive,
negative, or neutral. We develop two models: one for the regression task to predict the sentiment
score, and another for the classification task to predict the sentiment type.

## Dataset
The dataset consists of tweets and was collected from different researchers in the field of machine learning. Each tweet contains
some meta-information about e.g. the author, the number of likes, replies, etc., in addition to the text content
of the tweet itself.

The data is divided into 3 subsets, namely, a training set and two testing sets. The training set (n=8000)
is labeled to facilitate supervised learning. While the remaining two sets (n=1000 each) are unlabeled for
evaluating the final model. 

The labels here are, for the compound score, a continuous value between –1 and 1,
where –1 indicates a negative sentiment and 1 indicates the opposite. As for the sentiment label, it is a discrete
variable of either positive, negative, and neutral.

The total size of the dataset is about 4MB.

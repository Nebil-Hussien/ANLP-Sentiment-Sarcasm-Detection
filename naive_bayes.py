import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import make_pipeline
from dataloading import get_full_df, get_main_df, get_non_main_df
from evaluation import record_metrics
sns.set_theme() # use seaborn plotting style
    
train, test = get_full_df(2)
train_main = get_main_df()[0]
train_non_main = get_non_main_df(2)[0]

# original data
model1 = make_pipeline(CountVectorizer(), MultinomialNB())
model1.fit(train['tweet'].values.astype('U'), train['sarcastic'])

predictions1 = model1.predict(test['text'].values.astype('U'))

_, _, mat1 = record_metrics(test['sarcastic'], predictions1, "nb_full")

# all data
model2 = make_pipeline(CountVectorizer(), MultinomialNB())
model2.fit(train_main['tweet'].values.astype('U'), train_main['sarcastic'])

predictions2 = model2.predict(test['text'].values.astype('U'))

_, _, mat2 = record_metrics(test['sarcastic'], predictions2, "nb_main")

# all data without main
model3 = make_pipeline(CountVectorizer(), MultinomialNB())
model3.fit(train_non_main['tweet'].values.astype('U'), train_non_main['sarcastic'])

predictions3 = model3.predict(test['text'].values.astype('U'))

_, _, mat3 = record_metrics(test['sarcastic'], predictions3, "nb_non_main")

# all data without main with the same number of as main

model4 = make_pipeline(CountVectorizer(), MultinomialNB())
non_main_reduced = train_non_main.sample(frac=1).iloc[:train_main.shape[0]]
model4.fit(non_main_reduced['tweet'].values.astype('U'), non_main_reduced['sarcastic'])

predictions4 = model4.predict(test['text'].values.astype('U'))

_, _, mat4 = record_metrics(test['sarcastic'], predictions4, "nb_non_main_reduced")

# display confusion for three different set-ups
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
fig.supxlabel('true labels')
fig.supylabel('predicted labels')

titles = ['Original training set', 'Original-excluding training set', "Original-excluding scaled to original's size"]
for i, mat in enumerate([mat1, mat3, mat4]):
    group_counts = ["{0:0.0f}".format(value) for value in mat.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in mat.flatten()/np.sum(mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    mat = mat.copy()
    sns.heatmap(mat.T, ax=ax[i], square = True, annot=labels, fmt='', xticklabels=['sincere', 'sarcastic'], yticklabels=['sincere', 'sarcastic'], cbar=False)
    ax[i].set_title(titles[i], fontsize=8)

plt.show()
plt.close()

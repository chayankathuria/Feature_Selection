'''
According to wikipedia, "feature selection is the process of selecting a subset of relevent features for use in model construction". 
In normal circumstances, domain knowledge plays an important role.

Let's explore the following feature selection and dimensionality reduction techniques:

1. Remove features with missing values
2. Remove features with low variance
3. Remove highly correlated features
4. Univariate feature selection
5. Recursive feature elimination
6. Feature selection using SelectFromModel
7. PCA

'''
# Importing all the required modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Reading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Checking the shape
print('Train Shape: ', train.shape)
print('Test Shape: ', test.shape)

'''
Output-> 
Train Shape:  (250, 302)
Test Shape:  (19750, 301)

As it is clearly visible that the no. of features is too high compared to the number of instances.
We will have to significantly reduce the feature size in order to make a good model.

Moreover, the distribution of classes is 90:160 which is acceptable and not imbalanced.
'''

# Baseline Models

'''
We'll use logistic regression is a good baseline as it is fast to train and predict and scales well. 
We'll also use random forest. With its attribute feature_importances_ we can get a sense of which features are most important.
'''

# prepare for modeling
X_train_df = train.drop(['id', 'target'], axis=1)
y_train = train['target']

X_test = test.drop(['id'], axis=1)

# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test)

lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(n_estimators=100)

lr_scores = cross_val_score(lr,
                            X_train,
                            y_train,
                            cv=5,
                            scoring='roc_auc')
rfc_scores = cross_val_score(rfc, X_train, y_train, cv=5, scoring='roc_auc')

print('LR Scores: ', lr_scores)
print('RFC Scores: ', rfc_scores)

'''
Output->
LR Scores:LLR:[0.80729167 0.71875    0.734375   0.80034722 0.66319444]
RFC Scores:RFC:[0.59722222 0.6171875  0.66232639 0.75086806 0.69444444]
We can see the model is overfitting from the variation in cross validation scores. 
We can attempt to improve these scores through feature selection.
'''

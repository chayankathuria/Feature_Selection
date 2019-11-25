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
# 1. Remove features with missing values
'''
check for missing values and then can remove columns exceeding a threshold we define.
'''
# Checking for missing values; Will give true if even a single value in the whole df is missing. 
isnull().any().any()
'''
Output-> False
Hence no missing values
'''

# 2.  Remove features with low variance

'''
In sklearn's feature selection module we find VarianceThreshold.  It removes all features whose variance doesn't meet some threshold.  
By default it removes features with zero variance or features that have the same value for all samples.
'''
from sklearn import feature_selection

sel = feature_selection.VarianceThreshold()
train_variance = sel.fit_transform(train)
train_variance.shape

'''
We can see from above there are no features with the same value in all columns, so we have no features to remove here.  
We can revisit this technique later and consider removing features with low variance by changing the variance later.
'''

# 3. Remove highly correlated features
'''
Features that are highly correlated or colinear can cause overfitting.  Here we will explore correlations among features.
'''

# find correlations to target
corr_matrix = train.corr().abs()

print(corr_matrix['target'].sort_values(ascending=False).head(10))

# Select upper triangle of correlation matrix
matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
sns.heatmap(matrix)
plt.show;

# Find index of feature columns with high correlation
to_drop = [column for column in matrix.columns if any(matrix[column] > 0.50)]
print('Columns to drop: ' , (len(to_drop)))

'''
Output-> Columns to drop:  0

From the above correlation matrix we see that there are no highly correlated features in the dataset. 
And even exploring correlation to target shows feature 33 with the highest correlation of only 0.37.
'''




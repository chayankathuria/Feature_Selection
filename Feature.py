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

# 4. Univariate Feature Selection

'''
We can use sklearn's SelectKBest to select a number of top features to keep. 
This method uses statistical tests like the chi-square test to select features having the highest correlation to the target. 
Here we will keep the top 100 features.
'''

# feature extraction
k_best = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=100)
# fit on train set
fit = k_best.fit(X_train, y_train)
# transform train set
univariate_features = fit.transform(X_train)

univariate_features.shape # (250,100)

# Let's run the crossvalidation again to see the variance in scores

lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(n_estimators=100)

lr_scores = cross_val_score(lr, univariate_features, y_train, cv=5, scoring='roc_auc')
rfc_scores = cross_val_score(rfc, univariate_features, y_train, cv=5, scoring='roc_auc')

print('LR Scores: ', lr_scores)
print('RFC Scores: ', rfc_scores)

'''
Output->
LR Scores:  [0.89930556 0.93402778 0.89236111 0.96006944 0.94791667]
RFC Scores:  [0.78125    0.81336806 0.78038194 0.86545139 0.77690972]

Cross validation scores are improved as compared to the baseline above, 
but we still can see variation in the scores which indicates overfitting.
As visible, the variance is still high. As a low variance model like Logistic regression is varying almost by 6%.
'''

# 5. Recursive Feature Elimination

'''
Recursive feature selection works by eliminating the least important features. 
It continues recursively until the specified number of features is reached. 
Recursive elimination can be used with any model that assigns weights to features, either through coef_ or feature_importances_

Here we will use logistic regression to select the 100 best features.
'''

# feature extraction
rfe = feature_selection.RFE(lr, n_features_to_select=100)

# fit on train set
fit = rfe.fit(X_train, y_train)

# transform train set
recursive_features = fit.transform(X_train)

lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(n_estimators=10)

lr_scores = cross_val_score(lr, recursive_features, y_train, cv=5, scoring='roc_auc')
rfc_scores = cross_val_score(rfc, recursive_features, y_train, cv=5, scoring='roc_auc')

print('LR Scores: ', lr_scores)
print('RFC Scores: ', rfc_scores)

'''
Output->
LR Scores:  [0.99826389 0.99652778 0.984375   1.         0.99652778]
RFC Scores:  [0.71267361 0.66753472 0.71614583 0.55729167 0.60503472]

As evident, the logistic regression now is sufficiently robust and high performing. Let's check out other methods too.
'''

# 6. Feature selection using SelectFromModel

'''
Like recursive feature selection, sklearn's SelectFromModel is used with any estimator that has a coef_ or feature_importances_ attribute. 
It removes features with values below a set threshold
'''

# feature extraction
select_model = feature_selection.SelectFromModel(lr)

# fit on train set
fit = select_model.fit(X_train, y_train)

# transform train set
model_features = fit.transform(X_train)

lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(n_estimators=100)

lr_scores = cross_val_score(lr, model_features, y_train, cv=5, scoring='roc_auc')
rfc_scores = cross_val_score(rfc, model_features, y_train, cv=5, scoring='roc_auc')

print('LR Scores: ', lr_scores)
print('RFC Scores: ', rfc_scores)

'''
Output-> 
LR Scores:  [0.984375   0.99479167 0.97222222 0.99305556 0.99305556]
RFC Scores:  [0.86024306 0.82725694 0.75434028 0.88541667 0.80642361]

As can be seen, Logistic regression is still giving great results. But this time, the performance of rf also bumped up significantly.
'''

# 7. PCA

'''
PCA is not a feature selection/removal technique, but a feature extraction technique which uses all the features 
available and transform them into a lower dimensional space. This results in dimensionality reduction. The PCA Components now formed
are a set of totally new features formed from original features. Now, we can select top few components which explain most of the variance
in data. Hence we extracted features and reduced dimensionality.
'''

from sklearn.decomposition import PCA
# pca - keep 90% of variance
pca = PCA(0.90)

principal_components = pca.fit_transform(X_train)
principal_df = pd.DataFrame(data = principal_components)
principal_df.shape

'''
(250,139)
We can see that we are left with 139 features that explain 90% of the variance in our data
'''

lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(n_estimators=100)

lr_scores = cross_val_score(lr, principal_df, y_train, cv=5, scoring='roc_auc')
rfc_scores = cross_val_score(rfc, principal_df, y_train, cv=5, scoring='roc_auc')

print('LR Scores: ', lr_scores)
print('RFC Scores: ', rfc_scores)

'''
LR Scores:  [0.80902778 0.703125   0.734375   0.80555556 0.66145833]
RFC Scores:  [0.61284722 0.65798611 0.71614583 0.66927083 0.78125   ]
'''

# pca keep 75% of variance
pca = PCA(0.75)
principal_components = pca.fit_transform(X_train)
principal_df = pd.DataFrame(data = principal_components)
principal_df.shape

'''
(250,93)
'''

lr = LogisticRegression(solver='liblinear')
rfc = RandomForestClassifier(n_estimators=100)

lr_scores = cross_val_score(lr, principal_df, y_train, cv=5, scoring='roc_auc')
rfc_scores = cross_val_score(rfc, principal_df, y_train, cv=5, scoring='roc_auc')

print('LR Scores: ', lr_scores)
print('RFC Scores: ', rfc_scores)

'''
LR Scores:  [0.72048611 0.60069444 0.68402778 0.71006944 0.61284722]
RFC Scores:  [0.48958333 0.73003472 0.64322917 0.6328125  0.67013889]
'''

'''
From all the above techniques, RFE performed the best. Giving the best cv scores for lr and rfc.
'''

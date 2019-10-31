#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, KFold

from sklearn.metrics import recall_score, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score,                             classification_report, confusion_matrix


# In[4]:


from sklearn.linear_model import LogisticRegression


# ### Read data
# In the last three tutorials, I have processed data and finally selected some relevant features for the project. So let's read the data with selected features.

# In[5]:


df_selected = pd.read_csv('../Processed Data/df_processed_categorical_v3.csv')


# In[6]:


df_selected.describe(include = 'all')


# ### Class balance
# Before we jump into anything, we must take care of the class unbalance problems. The following code shows the number of examples in each class.

# In[7]:


df_selected.loan_status.value_counts(normalize=True)


# In[8]:


df_selected.loan_status.value_counts()


# Loan-status class is imbalanced. To solve the problem of unbalance class issue there are many techniques that can be applied. For example: 
# 
# (1) Assign a class weight (2) Use ensemble algorithms with cross-validation (3) Upsample minority class or downsample the majority class
# 
# I wrote a blog post describing the above three techniques. In this work, I have tried all the techniques and found upsampling minority class improves the model's generalization on unseen data. I the code below I upsample minority class with Scikit-learn ‘resample’ method.

# #### Upsample the minority class
# One of the popular techniques for dealing with highly unbalanced data sets is called resampling. Although the technique has proven to be effective in many cases to solve the unbalanced class issue, however, these techniques also have their weaknesses. For example, over-sampling records from the minority class, which can lead to overfitting while removing random records from the majority class, which can cause loss of information. Alright, now let's see how upsampling works better in this project:

# In[9]:


df_major = df_selected[df_selected.loan_status == 0]
df_minor = df_selected[df_selected.loan_status == 1]


# In[10]:


df_minor_upsmapled = resample(df_minor, replace = True, n_samples = 358436, random_state = 2018)


# In[11]:


df_minor_upsmapled = pd.concat([df_minor_upsmapled, df_major])


# In[12]:


df_minor_upsmapled.loan_status.value_counts()


# In the above code, I first separate classes into two data frames: 1. df_major and 2. df_minor. Then I use df_minor to upsample it to the same number as the major class which is 358436. Notice that I keep the replace option to true. If I were downsampled then I would keep the replace option to false. Finally, I concatenate the upsampled minor class with major class. Loom at the loan status value counts. They are the same now. Now it's time to standardize the data

# #### 0. Evaluate the model
# To see the performance of the unknown data, I wrote a function named as "evaluate_model" which prints different evaluation criteria: 1) accuracy, 2) ROC-AUC score, 3) confusion matrix and 4) detailed classification report.

# In[13]:


def evaluate_model(ytest, ypred, ypred_proba = None):
    if ypred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))
    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))


# #### 1. Standarize the data
# In this section, I summarize the data by removing the mean from each sample and then divide by the standard deviation. Zero mean and unit standard deviation helps the model’s optimization faster. I used the Scikit-learn StandardScaler method. Before that, I split the dataset into training and testing parts. The following code is self-explanatory:

# In[14]:


X = df_minor_upsmapled.drop('loan_status', axis = 1)
Y = df_minor_upsmapled.loan_status


# In[15]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)


# In[16]:


mms = StandardScaler()
mms.fit(xtrain)
xtrain_scaled = mms.transform(xtrain)


# In[17]:


np.shape(df_minor_upsmapled)


# Now that our data is ready, we can move on building models. I always start with a simple algorithm like logistic regression to keep things simple and record the performance as a benchmark for complex models.

# #### 2. logistic regression model
# 
# Logistic regression is a modeling technique borrowed from statistics. It is handier and go-to method for binary classification problems. As I said before the algorithm is relatively simple and easy to implement, I always first start with this technique and record the performance of the model for future complex model benchmarking purpose. It helps me move forward easily and intuitively. Alright, let's see how logistic regression can perform:

# In[18]:


logisticRegr = LogisticRegression()


# In[19]:


logisticRegr.fit(xtrain_scaled, ytrain)


# In the above code, I used default parameters. Bellow, I standardize the test data using the same standardization parameters (mean and standard deviation) used for training data.

# In[20]:


xtest_scaled = mms.transform(xtest)


# In[21]:


lr_pred = logisticRegr.predict(xtest_scaled)


# Finally, let's see the performance of the logistic regression:

# In[22]:


evaluate_model(ytest, lr_pred)


# The result is not promising. The accuracy of the model is just a little above the random guessing. We see that the simplest model gives 66% accuracy. Therefore, we have to pick a better algorithm and tune its hyperparameters in such a way that, the model outperforms the logistic regression model. 

# ## Choose an appropriate model
# 
# Choosing an appropriate model is another challenge for a data scientist. Sometimes even an experienced data scientist cannot tell which algorithm will perform the best before trying different algorithms. In our final dataset, almost 60% of our features are categorical. Therefore, a tree-based model may be a better choice. Still, it’s very unpredictable. If tree-based algorithms do not perform very well we might try another algorithm such as the neural network. In the project, I would try both bagging (Random forest)and boosted tree-based (LightGBM) algorithms. Alright, let's begin with random forest.

# ### 3. Random forest model
# Random Forest is a flexible, easy to use machine learning ensemble algorithm. The algorithom is so light and effective that even without hyper-parameter tuning, it can produce can great result. It is also one of the most used algorithms, because it’s simplicity and the fact that it can be used for both classification and regression tasks. Details of the method can be found on Scikit-sklearn webpage or in this blog post: https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd

# In[23]:


def random_forest(xtrain, xtest, ytrain):
    rf_params = {
        'n_estimators': 126, 
        'max_depth': 14
    }

    rf = RandomForestClassifier(**rf_params)
    rf.fit(xtrain, ytrain)
    rfpred = rf.predict(xtest)
    rfpred_proba = rf.predict_proba(xtest)
    
    return rfpred, rfpred_proba, rf


# In the above function, I first define the hyperparameters. The most important hyperparameters of random forest are the number of estimators and the maximum depth of a tree. I try to find the optimal hyperparameter value in the iterative process. I manually start with a small number of estimator then increase slowly. I find this manual process efficient and intuitive rather than using GridSearchCV or RandomSearch. There is another technique for Bayesian hyperparameter optimization, which can be used to find a suitable set of hyperparameters. The technique seems to be more efficient and effective. In my next project, I would try it. More details on this technique can be found in William Koehrsen's blog post "A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning". As I said before, I start with a low number with most important hyperparameter and once I find the optimum value I start with the next influential hyperparameter and so on. Alright, it's time to see the performance of random forest on test data sets:

# In[24]:


rfpred, rfpred_proba, rf = random_forest(xtrain_scaled, xtest_scaled, ytrain)


# In[25]:


evaluate_model(ytest, rfpred, rfpred_proba)


# Wow, Radom forest does better. Almost 11% accuracy jump from logistic regression. Which is a great achievement and proves that tree-based models perform well on categorical data. At this point, I stopped working on the random forest model as the model performance does not increase. I tried with other hyperparameters and increasing the n_estimators, that did not help. These difficulties helped me decide to use a gradient boosted tree. Since I plan not  to move further with Random forest, I must find out the robustness of the model. One way to find out the cross-validation. 

# ### Cross validation
# Cross-validation is one of the effective ways to assess a model and its generalization power using an independent data set in practice. If the model’s performance on different folds is consistent, then we could say that the model is robust and performing well. In the following, we test the RF model’s robustness using Scikit-sklearn cross-validation method:

# In[ ]:


scoring = ['accuracy', 'recall', 'roc_auc', 'f1']
scores = cross_validate(rf, X = xtrain_scaled, y = ytrain, scoring=scoring,
                         cv = 10, return_train_score = False, verbose = 10, n_jobs= -1)


# In the above code, I used four different evaluation metrics to judge the model's generalization. Let's see the result:

# In[31]:


scores


# The above code snippet, print out the results from the cross-validation. If you see different evaluation metrics carefully, you would see the model indeed is a robust and performs consistently across the folds. Let's do some more specific observations of the metric scores: mean and variance:

# In[32]:


print('F1 score# (1) mean: {} (2)variance: {}'.format(np.mean(scores['test_f1']), np.var(scores['test_f1'])))
print('Recall score# (1) mean: {} (2)variance: {}'.format(np.mean(scores['test_recall']), np.var(scores['test_recall'])))
print('Accuracy score# (1) mean: {} (2)variance: {}'.format(np.mean(scores['test_accuracy']), np.var(scores['test_accuracy'])))


# It's good to see that every evaluation metric has a very low variance which again confirms the model robustness. Although the model is robust, I am not happy yet. We need to improve the model generalization on testing data. Next, I would try a gradient boosted tree-based algorithm.
# 
# There are many gradients boosted tree-based algorithms available. For example XGBoost, LightGBM, CataBoost etc. I myself find LightGBM is faster and performs well on categorical data more than other algorithms I mentioned. So, let's get started with LightGBM.

# ### 4. LightGBM model

# In[33]:


import lightgbm


# In[34]:


lbg_params = {
    'n_estimators': 8000,
    'max_depth': 100,
    'objective': 'binary',
    'learning_rate' : 0.02,
    'num_leaves' : 250,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 1,
    'boosting_type' : 'gbdt'
}


# In[35]:


lgb = lightgbm.LGBMClassifier(**lbg_params)


# In[36]:


lgb.fit(xtrain_scaled, ytrain)


# In the above code, I first find the optimal hyperparameters manually similar way I did in the random forest model. I find the most important parameters are ‘n_estimators’ and ‘max_depth’ in the model. Let's see the model’s prediction and performance on the test data:

# #### Test the model with test data

# In[37]:


lgb_pred = lgb.predict(xtest_scaled)


# In[38]:


lgb_pred_proba = lgb.predict_proba(xtest_scaled)


# In[39]:


evaluate_model(ytest, lgb_pred, lgb_pred_proba)


# The performance report seems promising. Accuracy jumped 35% from logistic regression and 23% from the random forest model. I stop here optimizing other hyperparameters. It took me about 4 hours to find the above two hyperparameters. Now I focus on the model's robustness using same technique cross-validation.

# #### Cross validation

# In[40]:


folds = list(KFold(5, shuffle=True, random_state=2016)             .split(xtrain_scaled, ytrain))


# In[41]:


for i, (train_idx, valid_idx) in enumerate(folds):
    
    ytrain = np.array(ytrain)
    X_train = xtrain_scaled[train_idx]
    y_train = ytrain[train_idx]
    X_valid = xtrain_scaled[valid_idx]
    y_valid = ytrain[valid_idx]
    
    lgb.fit(X_train, y_train)
    pred = lgb.predict(X_valid)
    pred_proba = lgb.predict_proba(X_valid)
    
    print('\ncv: {}\n'.format(i))
    evaluate_model(y_valid, pred, pred_proba)


# If you look at the print out of the above code, you would see LightGBM is also robust and performs consistently across different training folds. In this project, I did not discuss anything about overfitting. Hope I will write another blog on the topic.


# Create the cross validation framework
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

""" Cross-validation framework: Stratified K-Fold cross-validation is created with k=5 (n_splits=5).
This means the dataset will be split into 5 folds, and in each iteration, one fold will be used 
for testing and the remaining folds for training the model. The shuffle parameter is set to False,
which means the data will not be shuffled before splitting. """
kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)


# Import the imbalance Learn module
""" 
Importing the imblearn pipeline and resampling modules: imblearn 
is a Python library for imbalanced learning, and it contains 
several techniques for balancing datasets, such as under-sampling 
and over-sampling. In this code, two techniques are imported: NearMiss 
for under-sampling and SMOTE for over-sampling. A pipeline is also 
created using these modules, which allows for chaining multiple
estimators together.
"""
from imblearn.pipeline import Pipeline ## creates a pipeline using the provided estimators.
from imblearn.under_sampling import NearMiss  ## perform Under-sampling  based on NearMiss methods. 
from imblearn.over_sampling import SMOTE ## PerformOver-sampling class that uses SMOTE.

# import the metrics  
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

# import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 
#      Building and Training the Model
# 
# =============================================================================
import ccfd_data_preparation as dp
"""
A script named "ccfd_data_preparation" is imported, which 
contains the preprocessed data split into training 
and testing sets. A RandomForestClassifier is chosen
as the model and is fitted on the training data, 
and the predictions are made on the testing data.
"""
# Fit and Predict
rfc = RandomForestClassifier()
rfc.fit(dp.X_train, dp.Y_train)
y_pred = rfc.predict(dp.x_test)


# For the performance let's use some metrics from SKLEARN module
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("The accuracy score is: ", accuracy_score(dp.y_test, y_pred))
print("The precision score is: ", precision_score(dp.y_test, y_pred))
print("The recall score is: ", recall_score(dp.y_test, y_pred))
print("The f1 score is: ", f1_score(dp.y_test, y_pred))

""" 
The accuracy score is:  0.9996137776061234
The precision score is:  0.975
The recall score is:  0.7959183673469388     
The f1 score is:  0.8764044943820225
"""
""" 
we had only 0.17% fraud transactions, and a model
predicting all transactions to be valid would 
have an accuracy of 99.83%. Luckily, 
our model exceeded that to over 99.96%.
"""
""" 
¤ Precision: 
        It is the total number of true positives divided by 
        the true positives and false positives. Precision makes
        sure we don't spot good transactions as fraudulent in our problem.
        
¤ Recall: 
        It is the total number of true positives divided by the true positives
        and false negatives. Recall assures we don't predict fraudulent transactions
        as all good and therefore get good accuracy with a terrible model.
        
¤ F1 Score: 
        It is the harmonic mean of precision and recall. It makes a good
        average between both metrics.
 """
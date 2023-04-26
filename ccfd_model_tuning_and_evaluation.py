
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
from imblearn.under_sampling import NearMiss  ## perform Under-sampling  based on NearMiss methods. 
from imblearn.over_sampling import SMOTE ## PerformOver-sampling class that uses SMOTE.

# import the metrics  
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score

# import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
"""
A script named "ccfd_data_preparation" is imported, which 
contains the preprocessed data split into training 
and testing sets. A RandomForestClassifier is chosen
as the model and is fitted on the training data, 
and the predictions are made on the testing data.
"""
''' # Fit and Predict
rfc = RandomForestClassifier()
rfc.fit(dp.X_train, dp.Y_train)
y_pred = rfc.predict(dp.x_test)


# For the performance let's use some metrics from SKLEARN module
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("The accuracy score is: ", accuracy_score(dp.y_test, y_pred))
print("The precision score is: ", precision_score(dp.y_test, y_pred))
print("The recall score is: ", recall_score(dp.y_test, y_pred))
print("The f1 score is: ", f1_score(dp.y_test, y_pred))
 '''
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
 
 # =============================================================================
# 
#      Undersampling | NearMiss Methods
# 
# =============================================================================

# perform undersampling on the training data

""" making a flexible function that can perform grid 
or randomized search on a given estimator and its 
parameters with or without under/oversampling and
returns the best estimator along with the performance
metrics: """

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score


def get_model_best_estimator_and_metrics(estimator, params, kf=kf, X_train=X_train, 
                                         y_train=y_train, X_test=X_test, 
                                         y_test=y_test, is_grid_search=True, 
                                         sampling=NearMiss(), scoring="f1", 
                                         n_jobs=2):
        
    """
    This function takes in the following parameters:
    
    Parameters:
    ----------
    estimator : object
        The machine learning estimator to be used.
    params : dict
        The hyperparameters to be tuned.
    kf : object
        The cross-validation framework.
    X_train : array-like
        The training data.
    y_train : array-like
        The target variable for the training data.
    X_test : array-like
        The testing data.
    y_test : array-like
        The target variable for the testing data.
    sampling : object, optional (default=None)
        The sampling technique to be used.
    scoring : str, optional (default='f1')
        The evaluation metric to be used.
    n_jobs : int, optional (default=2)
        The number of CPU cores to use for parallel computing.
    """
    if sampling is None:
        # make the pipeline of only the estimator, just so the remaining code will work fine
        pipeline = make_pipeline(estimator)
    else:
        # make the pipeline of over/undersampling and estimator
        pipeline = make_pipeline(sampling, estimator)
    # get the estimator name
    estimator_name = estimator.__class__.__name__.lower()
    # construct the parameters for grid/random search cv
    new_params = {f'{estimator_name}__{key}': params[key] for key in params}
    if is_grid_search:
        # grid search instead of randomized search
        search = GridSearchCV(pipeline, param_grid=new_params, cv=kf, scoring=scoring, return_train_score=True, n_jobs=n_jobs, verbose=2)
    else:
        # randomized search
        search = RandomizedSearchCV(pipeline, param_distributions=new_params, 
                                    cv=kf, scoring=scoring, return_train_score=True,
                                    n_jobs=n_jobs, verbose=1)
    # fit the model
    search.fit(X_train, y_train)
    cv_score = cross_val_score(search, X_train, y_train, scoring=scoring, cv=kf)
    # make predictions on the test data
    y_pred = search.best_estimator_.named_steps[estimator_name].predict(X_test)
    # calculate the metrics: recall, accuracy, F1 score, etc.
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_proba = search.best_estimator_.named_steps[estimator_name].predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    # return the best estimator along with the metrics
    return {
        "best_estimator": search.best_estimator_,
        "estimator_name": estimator_name,
        "cv_score": cv_score,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

import pandas as pd
# Cumulatively create a table for the ROC curve
## Create the dataframe
res_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

rfc_results = get_model_best_estimator_and_metrics(
    estimator=RandomForestClassifier(),
    params={
      'n_estimators': [50, 100, 200],
      'max_depth': [4, 6, 10, 12],
      'random_state': [13]
    },
    sampling=None,
    n_jobs=3,
)
res_table = res_table.append({'classifiers': rfc_results["estimator_name"],
                                        'fpr': rfc_results["fpr"], 
                                        'tpr': rfc_results["tpr"], 
                                        'auc': rfc_results["auc"]
                              }, ignore_index=True)


# %%
print(f"==={rfc_results['estimator_name']}===")
print("Model:", rfc_results['best_estimator'])
print("Accuracy:", rfc_results['accuracy'])
print("Recall:", rfc_results['recall'])
print("F1 Score:", rfc_results['f1_score'])

# %%
logreg_us_results = get_model_best_estimator_and_metrics(
    estimator=LogisticRegression(),
    params={"penalty": ['l1', 'l2'], 
                  'C': [ 0.01, 0.1, 1, 100], 
                  'solver' : ['liblinear']},
    sampling=NearMiss(),
    n_jobs=3,
)
print(f"==={logreg_us_results['estimator_name']}===")
print("Model:", logreg_us_results['best_estimator'])
print("Accuracy:", logreg_us_results['accuracy'])
print("Recall:", logreg_us_results['recall'])
print("F1 Score:", logreg_us_results['f1_score'])
res_table = res_table.append({'classifiers': logreg_us_results["estimator_name"],
                                        'fpr': logreg_us_results["fpr"], 
                                        'tpr': logreg_us_results["tpr"], 
                                        'auc': logreg_us_results["auc"]
                              }, ignore_index=True)
res_table

# %%
# Plot the ROC curve for undersampling
res_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(17,7))

for j in res_table.index:
    plt.plot(res_table.loc[j]['fpr'], 
             res_table.loc[j]['tpr'], 
             label="{}, AUC={:.3f}".format(j, res_table.loc[j]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Positive Rate(False)", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("Positive Rate(True)", fontsize=15)
plt.title('Analysis for Oversampling', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()

# %%
# Cumulatively create a table for the ROC curve
res_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

lin_reg_os_results = get_model_best_estimator_and_metrics(
    estimator=LogisticRegression(),
    params={"penalty": ['l1', 'l2'], 'C': [ 0.01, 0.1, 1, 100, 100], 
            'solver' : ['liblinear']},
    sampling=SMOTE(random_state=42),
    scoring="f1",
    is_grid_search=False,
    n_jobs=2,
)
print(f"==={lin_reg_os_results['estimator_name']}===")
print("Model:", lin_reg_os_results['best_estimator'])
print("Accuracy:", lin_reg_os_results['accuracy'])
print("Recall:", lin_reg_os_results['recall'])
print("F1 Score:", lin_reg_os_results['f1_score'])
res_table = res_table.append({'classifiers': lin_reg_os_results["estimator_name"],
                                        'fpr': lin_reg_os_results["fpr"], 
                                        'tpr': lin_reg_os_results["tpr"], 
                                        'auc': lin_reg_os_results["auc"]
                              }, ignore_index=True)

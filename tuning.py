import numpy as np

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV


PARAMETERS_HGBC ={
    "learning_rate":[0.1, 0.2, 0.3],
    "max_iter": [150, 200],
    "max_leaf_nodes": [20,30,40,None],
    "min_samples_leaf":[ 2, 4, 10],
    "l2_regularization" : [0, 0.1, 0.2, 0.3],
    "early_stopping": [True, False]
}


PARAMETERS_RF= {
    "max_features" : ["sqrt", "log2"], 
    "criterion": ["gini", "entropy"],
    "min_samples_leaf":np.arange(10,70,10),
    "min_samples_split": np.arange(2, 10, 1)  
}


def tuning_classifiers(clf, parameters_grid, X_train, y_train, k_fold = 3, normal_grid_search = False) -> None:
    # Tune classifiers with Halving Search CV if "normal_grid_search" is false otherwise use Grid Search CV
    if normal_grid_search: 
        cv = GridSearchCV(clf, parameters_grid, cv=k_fold, scoring="f1_macro", n_jobs=-1).fit(X_train,y_train)
    else:
        cv =  HalvingGridSearchCV(clf, parameters_grid, cv=k_fold, scoring="f1_macro", n_jobs=-1, random_state = 42).fit(X_train,y_train)

    print(cv.best_estimator_)
    print(cv.best_score_)
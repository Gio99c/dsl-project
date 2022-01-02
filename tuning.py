import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from preprocessing import create_hours_bins


bool_par = [True, False]


# RANDOM FOREST

parameters_random_forest = {
    "n_estimators" : np.arange(start=500, stop = 5000, step = 500), 
    "max_features" : ["sqrt", "log2"], 
    "max_depth ": np.arange(start= 10, stop = 30, step = 5), 
    "criterion": ["gini", "entropy"],
    "n_jobs": -1, 
    "min_samples_split" :[2, 5, 10],
    "min_samples_leaf":[1, 2, 4],
    "bootstrap" : bool_par
}

# SUPPORT VECTOR MACHINE 

parameters_support_vector ={
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],
    "degree": np.arange(start=1, stop= 10, step = 1 ),
    "verbose": bool_par,
    "decision_function_shape": ['ovo', 'ovr'], 
    "break_ties": bool_par,
    "probability": bool_par, 
    "shrinking": bool_par,
    "gamma":['scale', 'auto'] 
    }

# NAIVE BAYES

parameters_naive_bayesian = {

    "priors": [np.array([x, 1-x]) for x in np.arange(0.01, 1, 0.02)], # Array prior probabilty 
    "var_smoothing": np.logspace(0.-12, num=150)

}

# MULTI LAYER NEURAL NETWORK 

parameters_multi_layer_nn = {

}


x = np.arange(0.01, 1, 0.02)
print(x)
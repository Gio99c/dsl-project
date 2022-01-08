import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier



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




parameters_HGBC ={
    "loss": ["auto", "binary_crossentropy", "categorical_crossentropy"],
    "learning_rate": np.arange(start=0.1, stop=1, step=0.1),
    "max_iter": np.arange(start=100, stop= 1000, step= 100),
    "max_leaf_nodes": [20,25,30,35,50,None],
    "min_samples_leaf":[10, 20, 40],
    "l2_regularization" : np.arange(start=0, stop=1, step=0.1),
    "warm_start": [True, False]

}


def test_classifier(clf, X_train : pd.DataFrame, X_test : pd.DataFrame, y_train: pd.Series, file_name: str) -> None:

    print(f'Start Training with {clf}')
    clf.fit(X_train,y_train)
    print(f'Finish Training with {clf}')

    y_pred = clf.predict(X_test)

    pd.Series(y_pred, name="Predicted").to_csv(file_name, index_label="Id")
    print(f"File Salvato con questo nome: {file_name}!\n\n\n")
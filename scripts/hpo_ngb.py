"""
=================
2.3 hpo ngboost
=================
This file shows how to optimize hyperparameters of ngboost model.
"""

from typing import Union

import os
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal, LogNormal

from ai4water import Model
from ai4water.utils import TrainTestSplit
from ai4water.utils.utils import get_version_info
from ai4water.utils.utils import dateandtime_now, jsonize
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer

from utils import read_data

# %%
for lib,ver in get_version_info().items():
    print(lib, ver)

# %%

data = read_data(target='Area (ABD) Mean')

input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]

print(input_features)

# %%
print(output_features)

# %%
# split the data into training and test. The **test data will not be used druing hpo**.

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(
    data[input_features],
    data[output_features]
)

print(TrainX.shape, TestX.shape, TrainY.shape, TestY.shape)

# %%

DISTS = {
    "Normal": Normal,
    "LogNormal": LogNormal,
    "Exponential": Exponential
}

ITER = 0
VAL_SCORES = []
SUGGESTIONS = []
num_iterations = 150  # number of hyperparameter iterations
SEP = os.sep
PREFIX = f"hpo_{dateandtime_now()}"  # folder name where to save the results
algorithm = "bayes"

# %%
# define parameter space
param_space = [
    Categorical(["Normal", "LogNormal", "Exponential"], name="Dist"),
    Integer(100, 1000, name="n_estimators"),
    Real(0.001, 0.5, name="learning_rate"),
    #Real(0.4, 1.0, name="minibatch_frac"),
    #Real(0.4, 1.0, name="col_sample")
]

# %%
# initial values of hyperparameters
x0 = ["Normal",
    100, 0.01, #1.0, 1.0
      ]

# %%
# define objective function

def objective_fn(
        return_model:bool = False,
        **suggestions
)->Union[float, Model]:
    """
    The output of this function will be minimized
    :param return_model: whether to return the trained model or the validation
        score. This will be set to True, after we have optimized the hyperparameters
    :param suggestions: contains values of hyperparameters at each iteration
    :return: the scalar value which we want to minimize. If return_model is True
        then it returns the trained model
    """
    global ITER

    suggestions = jsonize(suggestions)
    SUGGESTIONS.append(suggestions)
    dist = suggestions.pop("Dist")

    # build the model
    ngb = NGBRegressor(Dist=DISTS[dist],
                       verbose=False,
                       **suggestions)
    model = Model(
        model=ngb,
        mode="regression",
        category="ML",
        cross_validator={"KFold": {"n_splits": 5}},
        input_features=input_features,
        output_features=output_features,
        verbosity=-1
    )

    if return_model:
        model.fit(TrainX.values, TrainY.values,
                  validation_data=(TestX, TestY.values))
        model.evaluate(TestX, TestY, metrics=["r2", "r2_score"])
        return model

    # get the cross validation score which we will minimize
    val_score_ = model.cross_val_score(TrainX.values, TrainY.values)[0]

    # since cross val score is r2_score, we need to subtract it from 1. Because
    # we are interested in increasing r2_score, and HyperOpt algorithm always
    # minizes the objective function
    val_score = 1 - val_score_

    VAL_SCORES.append(val_score)
    best_score = round(np.nanmin(VAL_SCORES).item(), 2)
    bst_iter = np.argmin(VAL_SCORES)

    ITER += 1

    print(f"{ITER} {round(val_score, 2)} {round(val_score_, 2)}. Best was {best_score} at {bst_iter}")

    return val_score

# %%
# initialize the hpo
optimizer = HyperOpt(
    algorithm=algorithm,
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=False,  # we can turn it False if we want post-processing of results
    opt_path=f"results{SEP}{PREFIX}"
)

# %%
# run the hpo

# res = optimizer.fit()

# %%
# print optimized hyperparameters

# print(optimizer.best_paras())

# %%
# plot convergence

# optimizer.plot_convergence(show=True)

# %%

# optimizer.plot_convergence(original=True, show=True)

# %%
# plot explored hyperparameters as explored during hpo

# optimizer.plot_parallel_coords(show=True)

# %%
# build and train the model with optimized hyperparameters

# best_model = objective_fn(return_model=True, **optimizer.best_paras())

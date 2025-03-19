"""
==========================================
4.1 Bayesian NN (Disinfection Efficiency)
==========================================
This file shows how to record epistemic uncertainty in a neural network
for modeling Cell Count data.
"""

import numpy as np

from easy_mpl import plot

import matplotlib.pyplot as plt

from SeqMetrics import RegressionMetrics

from ai4water.utils import TrainTestSplit
from ai4water.postprocessing import ProcessPredictions

from utils import SAVE
from utils import read_data, BayesModel, version_info
from utils import set_rcParams, residual_plot, regression_plot

# %%
for lib, ver in version_info().items():
    print(lib, ver)

# %%
set_rcParams()

# %%
data = read_data()

input_features = data.columns.tolist()[0:-1]
output_features = data.columns.tolist()[-1:]


TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(
    data[input_features],
    data[output_features]
)

print(TrainX.shape, TestX.shape, TrainY.shape, TestY.shape)

# %%
# hyperparameters
# ----------------

hidden_units = [5, 5]
learning_rate = 0.00472268229046
activation = "elu"
train_size = len(TrainX)

num_epochs = 5000
batch_size = 32

# %%
# Build model
# -------------

model = BayesModel(
    model = {"layers": dict(hidden_units=hidden_units,
                            train_size=train_size,
                            activation=activation
                            )},
    batch_size=batch_size,
    epochs=num_epochs,
    lr=learning_rate,
    input_features=input_features,
    output_features=output_features,
    category= "DL",
    y_transformation="robust",
    optimizer="RMSprop",
    x_transformation=[
        {"method": "log2", "features": ["Time (min)"], "replace_zeros": True},
        {"method": "quantile", "features": ["Ini. CC"]},
        #{"method": "log2", "features": ["sonic_pd"]},
        {"method": "quantile", "features": ["h20 Conc."]},
        {"method": "quantile", "features": ["Volume (mL)"]},
        {"method": "log10", "features": ["Solution pH"]},
    ]
    #wandb_config=dict(project="flowcam", entity="atherabbas", monitor="val_loss")
)

# %%
# model training
model.fit(TrainX, TrainY, validation_data=(TestX, TestY),
          verbose=0)

# %%
# training results
# -------------

train_predictions = []
for i in range(100):
    train_predictions.append(model.predict(TrainX, verbose=0))
train_predictions =  np.concatenate(train_predictions, axis=1)

train_std = np.std(train_predictions, axis=1)
train_mean = np.mean(train_predictions, axis=1)

metrics = RegressionMetrics(TrainY, train_mean)
print(f"R2: {metrics.r2()}")
print(f"R2 Score: {metrics.r2_score()}")
print(f"RMSE Score: {metrics.rmse()}")
print(f"MAE: {metrics.mae()}")
# %%
processor = ProcessPredictions(
    mode="regression", forecast_len=1,
    path=model.path
)

# %%
processor.edf_plot(TrainY, train_mean)

# %%

plot(train_mean, '.', label="Prediction Mean", show=False)
plot(TrainY.values, '.', label="True", ax_kws=dict(logy=True))


# %%
# test results
# -------------

test_predictions = []
for i in range(100):
    test_predictions.append(model.predict(TestX, verbose=0))

test_predictions =  np.concatenate(test_predictions, axis=1)

print(test_predictions.shape)

# %%
test_std = np.std(test_predictions, axis=1)
test_mean = np.mean(test_predictions, axis=1)

metrics = RegressionMetrics(TestY, test_mean)
print(f"R2: {metrics.r2()}")
print(f"R2 Score: {metrics.r2_score()}")
print(f"RMSE Score: {metrics.rmse()}")
print(f"MAE: {metrics.mae()}")


# %%
processor.edf_plot(TestY, test_mean)

# %%

if model.use_wb:
    model.wb_finish()

# %%

residual_plot(
    TrainY.values,
    train_mean,
    TestY.values,
    test_mean,
    #label="Cell Count"
)
if SAVE:
    plt.savefig("results/figures/residue_bayes_eff", dpi=600, bbox_inches="tight")
plt.show()

# %%
ax = regression_plot(
    TrainY.values, train_mean,
    TestY.values, test_mean,
    max_ticks=None,
    label="Efficiency (%)"
)
ax.set_xlim([-2, 100])
ax.set_ylim([-2, 100])
if SAVE:
    plt.savefig("results/figures/reg_bayes_eff", dpi=600, bbox_inches="tight")
plt.show()

# %%

lower = np.min(test_predictions, axis=1)
upper = np.max(test_predictions, axis=1)
_, ax = plt.subplots(figsize=(6, 3))
ax.fill_between(np.arange(len(lower)), upper, lower, alpha=0.5, color='C1')
p1 = ax.plot(test_mean, color="C1", label="Prediction")
p2 = ax.fill(np.NaN, np.NaN, color="C1", alpha=0.5)
plt.show()

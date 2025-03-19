"""
=========================
5. Comparative Analysis
=========================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from easy_mpl import taylor_plot

from ai4water.utils import edf_plot

from utils import set_rcParams, version_info, SAVE

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%
# plotting empirical distribution function of absolute error
# between true and predicted values
set_rcParams()

# %%

LABELS = {
    'ngb': 'NGBoost',
    'aleoteric': "Bayesian",
    'area': 'Area',
    'cell_count': 'Disinfection Efficiency (%)',
    'train': 'Training',
    'test': 'Test'
}

# %%

obs = {}
sim = {
    "train_cell_count": {},
    "test_cell_count": {},
    "train_area": {},
    "test_area": {},
}

# %%

_, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 5), sharey="all")
ax.grid(visible=True, ls='--', color='lightgrey')
ax2.grid(visible=True, ls='--', color='lightgrey')

for model in ['ngb', 'aleoteric']:
    for target in ['cell_count', 'area']:
        for mode in ['train', 'test']:

            fpath = f"results/{model}_{target}/{mode}.csv"
            df = pd.read_csv(fpath)
            print(model, target, mode, df.sum())

            obs[f"{mode}_{target}"] = df.iloc[:, 0].values.reshape(-1,)
            sim[f"{mode}_{target}"][model] = df.iloc[:, 1].values.reshape(-1,)

            color = '#005066' if model == 'ngb' else '#B3331D'
            label = f"{LABELS[model]} ({LABELS[mode]})"
            linestyle = '-' if mode == "train" else ':'

            error = np.abs(df.iloc[:, 0] - df.iloc[:, 1])

            if target == "cell_count":
                edf_plot(error, linestyle=linestyle,
                         label=label, color=color,
                          ax=ax, show=False)
            else:
                edf_plot(error, linestyle=linestyle,
                         label=label, color=color,
                          ax=ax2, show=False)
ax.legend(loc=(0.35, 0.05), frameon=False)
ax2.legend(loc=(0.35, 0.05), frameon=False)
ax2.set_xlabel('Absolute Error')
ax2.set_ylabel('')
ax.set_xlabel('Absolute Error')
ax.set_title('Disinfection Efficiency', fontsize=12, weight="bold")
ax2.set_title('Area', fontsize=12, weight="bold")
if SAVE:
    plt.savefig("results/figures/edf", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

figure = taylor_plot(
    observations=obs,
    simulations=sim,
    plot_bias=True,
    show=False,
    figsize =(11, 8),
)
figure.axes[0].axis['left'].label.set_text('')
figure.axes[1].axis['left'].label.set_text('')
figure.axes[0].set_title('Disinf. Eff. (%) (Train)', fontsize=14, weight="bold")
figure.axes[1].set_title('Disinf. Eff. (%) (Test)', fontsize=14, weight="bold")
figure.axes[2].set_title('Area (Train)', fontsize=14, weight="bold")
figure.axes[3].set_title('Area (Test)', fontsize=14, weight="bold")

figure.legends[0].get_texts()[1].set_text('NGBoost')
figure.legends[0].get_texts()[2].set_text('Bayesian')

figure.axes[0].ticklabel_format(axis='x', style='sci', scilimits=(0,0))

if SAVE:
    plt.savefig("results/figures/taylor", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

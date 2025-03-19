"""
==============================
1. Exploratory Data Analysis
==============================


"""

import matplotlib.pyplot as plt

from ai4water.eda import EDA
from ai4water.utils import TrainTestSplit

from easy_mpl import hist
from easy_mpl import plot
from easy_mpl import scatter
from easy_mpl import boxplot
from easy_mpl.utils import create_subplots
from easy_mpl.utils import map_array_to_cmap, process_cbar

from utils import read_data
from utils import COLUMN_MAPS
from utils import set_rcParams, version_info

# %%

for k,v in version_info().items():
    print(k, v)

# %%

set_rcParams()

# %%

COLUMN_MAPS_ = {v:k for k,v in COLUMN_MAPS.items()}
COLUMN_MAPS_['ww_conc'] = "Wastewater Conc."
COLUMN_MAPS_['sonic_pd'] = "Sonicator Power"
COLUMN_MAPS_['h20_conc.'] = 'H2O2 Conc.'

# %%

data = read_data()
data_area = read_data(target='Area (ABD) Mean')
data_both = read_data(target=['Area (ABD) Mean', 'Efficiency'])

print(data.shape)

# %%

data_both.describe()

# %%

eda = EDA(data=data, save=False, show=False)

ax = eda.correlation(figsize=(8,8), square=True,
                     cbar_kws={"shrink": .72},
                     cmap="Spectral"
                     )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold', rotation=70)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

# %%
# Correlation
# ===============

eda = EDA(data=data_area, save=False, show=False)

ax = eda.correlation(figsize=(8,8), square=True,
                     cbar_kws={"shrink": .72},
                     cmap="Spectral"
                     )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold', rotation=70)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

# %%


eda = EDA(data=data_both, save=False, show=False)

ax = eda.correlation(figsize=(8,8), square=True,
                     cbar_kws={"shrink": .72},
                     cmap="Spectral"
                     )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold', rotation=70)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

# %%
# Distribution
# ==============

f, axes = create_subplots(data_both.shape[1], sharex="all")

for col, ax in zip(data_both.columns, axes.flatten()):

    boxplot(
        data_both[col], labels=col, ax=ax, show=False,
        fill_color="lightpink",
        patch_artist=True,
        widths=0.7,
        flierprops=dict(ms=2.0),
        medianprops={"color": "black"},
            )

    ax.set_xlabel(col)

plt.tight_layout()
plt.show()

# %%

train_count, test_count, _, _ = TrainTestSplit(seed=313).split_by_random(
    data['Efficiency'],
)

ax, _ = boxplot([train_count, test_count],
                flierprops=dict(ms=2.0),
                widths=0.6,
                labels=["Train", "Test"],
                showmeans=True,
                patch_artist=True,
                fill_color=["darkorange", "peachpuff"],
                medianprops={"color": "black", 'linewidth': 2},
                capprops={"linewidth":2}, whiskerprops=dict(linewidth=2),
                line_width=2.,
                meanprops={"markerfacecolor": "black",
                           "markeredgecolor": 'black',
                           "marker": "o"},
                show=False
        )
ax.tick_params(labelsize=12)
ax.set_xticklabels(["Train", "Test"], fontsize=12)
ax.grid(visible=True, ls='--', color='lightgrey')
plt.show()
# %%


_ = hist([train_count.values, test_count.values],
     labels=["Train", "Test"], alpha=0.7)

# %%
train_area, test_area, _, _ = TrainTestSplit(seed=313).split_by_random(
    data_area['Area (ABD) Mean'],
)

ax, _ = boxplot([train_area, test_area],
                flierprops=dict(ms=2.0),
                widths=0.6,
                labels=["Train", "Test"],
                showmeans=True,
                patch_artist=True,
                fill_color=["darkorange", "peachpuff"],
                medianprops={"color": "black", 'linewidth': 2},
                capprops={"linewidth":2}, whiskerprops=dict(linewidth=2),
                line_width=2.,
                meanprops={"markerfacecolor": "black",
                           "markeredgecolor": 'black',
                           "marker": "o"},
               show=False
               )
ax.tick_params(labelsize=12)
ax.set_xticklabels(["Train", "Test"], fontsize=12)
ax.grid(visible=True, ls='--', color='lightgrey')
plt.show()

# %%
_ = hist([train_area.values, test_area.values],
     labels=["Train", "Test"], alpha=0.7)

# %%
# line plot
# ===========

fig, axes = create_subplots(data_both.shape[1])

for ax, col, label  in zip(axes.flat, data_both, data_both.columns):

    plot(data_both[col].values, ax=ax,
         ax_kws=dict(ylabel=COLUMN_MAPS_.get(col, col),
                     ylabel_kws={"fontsize": 12, 'weight': 'bold'},),
         lw=0.9,
         color='darkcyan', show=False)
plt.tight_layout()
plt.show()

# %%
# Feature Interaction
# ====================

def draw_scatter(target, ax, label="Efficiency"):
    ax.grid(visible=True, ls='--', color='lightgrey')
    c, mapper = map_array_to_cmap(data[target].values, "inferno")

    if target in ["Sonic. PD", "Volume (mL)"]:
        ylabel = None
    else:
        ylabel = label

    if target in ['Solution pH', 'Volume (mL)']:
        xlabel = "Time (min)"
    else:
        xlabel = None

    ax_, _ = scatter(data_both['Time (min)'], data_both[label],
                      color=c, alpha=0.5, s=40, ec="grey", zorder=10,
                      ax_kws=dict(logy=True, ylabel=ylabel,
                                  ylabel_kws={"fontsize": 12, 'weight': 'bold'},
                                  top_spine=False, right_spine=False,
                                  xlabel=xlabel,
                                  xlabel_kws={"fontsize": 12, 'weight': 'bold'}),
                      ax=ax, show=False)
    process_cbar(ax_, mappable=mapper, orientation="vertical", pad=0.1,
                 border=False,
                 title=COLUMN_MAPS_.get(target, target),
                 title_kws=dict(fontsize=12))
    return


f, all_axes = create_subplots(5, sharex="all", facecolor="#EFE9E6", figsize=(9, 6))

targets = ['Ini. CC', 'Sonic. PD', 'h20 Conc.', 'Volume (mL)', 'Solution pH']
for col, axes in zip(targets, all_axes.flatten()):
    draw_scatter(col, axes)

plt.tight_layout()
plt.show()

# %%
f, all_axes = create_subplots(5, sharex="all", facecolor="#EFE9E6", figsize=(9, 6))

targets = ['Ini. CC', 'Sonic. PD', 'h20 Conc.', 'Volume (mL)', 'Solution pH']
for col, axes in zip(targets, all_axes.flatten()):
    draw_scatter(col, axes, label="Area (ABD) Mean")

plt.tight_layout()
plt.show()

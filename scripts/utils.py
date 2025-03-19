"""
==============
utils
==============
This file contains utility functions which are used
in other files.
"""
import sys
import warnings
from typing import Union, Any, List

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import shap
import ngboost

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import layers

from easy_mpl import scatter, regplot, plot, hist
from easy_mpl.utils import AddMarginalPlots  # to add marginal plots along an axes

from SeqMetrics import RegressionMetrics

from ai4water.utils.utils import get_version_info
from ai4water.functional import Model as FModel

# %%

SAVE = False

# %%

COLUMN_MAPS = {
'Wastewater concentration (Ci)': "WW Conc.",
'Cyanobacterial cell count': 'Ini. CC',
'Sonicator power density':'Sonic. PD',
'Concentration of H2O2':'h20 Conc.',
#'Solution pH': 'sol_ph'
}

COLUMN_MAPS_ = {v:k for k,v in COLUMN_MAPS.items()}
COLUMN_MAPS_['ww_conc'] = "Wastewater Conc."
COLUMN_MAPS_['sonic_pd'] = "Sonicator Power"
COLUMN_MAPS_['h20_conc.'] = 'H2O2 Conc.'
COLUMN_MAPS_['ini_cc'] = "Ini. Cell Count"

# %%

def read_data(
        inputs:List[str]=None,
        target = None
)->pd.DataFrame:

    df = pd.read_csv("data_0315.csv")

    if inputs is None:
        inputs = ['Time (min)', 'Cyanobacterial cell count', #'Wastewater concentration (Ci)',
                  'Sonicator power density', 'Concentration of H2O2', 'Volume (mL)',
                  'Solution pH']

    # calculate efficiency and put it in dataframe
    ini_cell_count = df['Cyanobacterial cell count'].values
    fin_cell_count = df['final count/mL'].values
    efficiency = ((ini_cell_count - fin_cell_count) / ini_cell_count) * 100
    efficiency = np.where(efficiency<0.0, 0.0, efficiency)

    df['Efficiency'] = efficiency

    if target is None:

        target = "Efficiency"

    if not isinstance(target, list):
        target = [target]

    columns = inputs + target

    df = df[columns]

    df = df.rename(columns=COLUMN_MAPS)
    return df

# %%
def prior(kernel_size, bias_size, dtype=None):
    # Define the prior weight distribution as Normal of mean=0 and stddev=1.
    # Note that, in this example, the we prior distribution is not trainable,
    # as we fix its parameters.
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# %%

def posterior(kernel_size, bias_size, dtype=None):
    # Define variational posterior weight distribution as multivariate Gaussian.
    # Note that the learnable parameters for this distribution are the means,
    # variances, and covariances.
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# %%

class BayesModel(FModel):
    """
    A model which can be used to quantify aleotoric uncertainty, or epsitemic
    uncertainty or both. Following parameters must be defined in a dictionary
    called ``layers``.
    >>> model = BayesModel(model={"layers": {'hidden_units': [1,], 'train_size': 100,
    ...           'activation': 'sigmoid'}})


    hidden_units : List[int]
    train_size : int
    activation : str
    uncertainty_type : str
        either ``epistemic`` or ``aleoteric`` or ``both``
    """
    def add_layers(self, *args, **kwargs)->tuple:
        hidden_units = self.config['model']['layers']['hidden_units']
        train_size = self.config['model']['layers']['train_size']
        activation = self.config['model']['layers']['activation']
        uncertainty_type = self.config['model']['layers'].get('uncertainty_type',
                                                              'epistemic')

        assert uncertainty_type in ("epistemic", "aleoteric", "both")
        epistemic = False
        aleoteric = False

        if uncertainty_type in ("epistemic", "both"):
            epistemic = True

        if uncertainty_type in ("aleoteric", "both"):
            aleoteric = True

        inputs = layers.Input(shape=len(self.input_features, ), dtype=tf.float32)
        features = layers.BatchNormalization()(inputs)

        if epistemic:
            # Create hidden layers with weight uncertainty using the
            # DenseVariational layer.
            for units in hidden_units:
                features = tfp.layers.DenseVariational(
                    units = units,
                    make_prior_fn = prior,
                    make_posterior_fn = posterior,
                    kl_weight = 1 / train_size,
                    activation = activation,
                )(features)
        else:
            for units in hidden_units:
                features = layers.Dense(units, activation=activation)(features)

        if aleoteric:
            # Create a probabilisticå output (Normal distribution), and use
            # the `Dense` layer
            # to produce the parameters of the distribution.
            # We set units=2 to learn both the mean and the variance of the
            # Normal distribution.
            distribution_params = layers.Dense(units=2)(features)
            outputs = tfp.layers.IndependentNormal(1)(distribution_params)
        else:
            # The output is deterministic: a single point estimate.
            outputs = layers.Dense(units=1)(features)

        return inputs, outputs

# %%

def shap_scatter(
        feature_shap_values:np.ndarray,
        feature_data:Union[pd.DataFrame, np.ndarray, pd.Series],
        color_feature:pd.Series=None,
        color_feature_is_categorical:bool = False,
        feature_name:str = '',
        show_hist:bool = True,
        palette_name = "tab10",
        s:int = 70,
        ax:plt.Axes = None,
        edgecolors='black',
        linewidth=0.8,
        alpha=0.8,
        show:bool = True,
        **scatter_kws,
):
    """

    :param feature_shap_values:
    :param feature_data:
    :param color_feature:
    :param color_feature_is_categorical:
    :param feature_name:
    :param show_hist:
    :param palette_name:
        only relevant if ``color_feature_is_categorical`` is True
    :param s:
    :param ax:
    :param edgecolors:
    :param linewidth:
    :param alpha:
    :param show:
    :param scatter_kws:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()

    if color_feature is None:
        c = None
    else:
        if color_feature_is_categorical:
            if isinstance(palette_name, (tuple, list)):
                assert len(palette_name) == len(color_feature.unique())
                rgb_values = palette_name
            else:
                rgb_values = sns.color_palette(palette_name, color_feature.unique().__len__())
            color_map = dict(zip(color_feature.unique(), rgb_values))
            c= color_feature.map(color_map)
        else:
            c = color_feature.values.reshape(-1,)

    _, pc = scatter(
        feature_data,
        feature_shap_values,
        c=c,
        s=s,
        marker="o",
        edgecolors=edgecolors,
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
        show=False,
        **scatter_kws
    )

    if color_feature is not None:
        feature_wrt_name = ' '.join(color_feature.name.split('_'))
        if color_feature_is_categorical:
            # add a legend
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                              label=k, markersize=8) for k, v in color_map.items()]

            ax.legend(title=feature_wrt_name,
                  handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                      title_fontsize=14
                      )
        else:
            cbar = plt.colorbar(pc, aspect=80)
            cbar.ax.set_ylabel(feature_wrt_name, rotation=90, labelpad=14,
                               fontsize=14, weight="bold")

            set_yticklabels(cbar.ax, max_ticks=None)

            cbar.set_alpha(1)
            cbar.outline.set_visible(False)

    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"SHAP value for {feature_name}")
    ax.axhline(0, color='grey', linewidth=1.3, alpha=0.3, linestyle='--')


    set_xticklabels(ax, max_ticks=None)
    set_yticklabels(ax, max_ticks=None)

    if show_hist:
        if isinstance(feature_data, (pd.Series, pd.DataFrame)):
            feature_data = feature_data.values
        x = feature_data

        if len(x) >= 500:
            bin_edges = 50
        elif len(x) >= 200:
            bin_edges = 20
        elif len(x) >= 100:
            bin_edges = 10
        else:
            bin_edges = 5

        ax2 = ax.twinx()

        xlim = ax.get_xlim()

        ax2.hist(x.reshape(-1,), bin_edges,
                 range=(xlim[0], xlim[1]),
                 density=False, facecolor='#000000', alpha=0.1, zorder=-1)
        ax2.set_ylim(0, len(x))
        ax2.set_yticks([])

    if show:
        plt.show()

    return ax

# %%

def set_xticklabels(
        ax:plt.Axes,
        max_ticks:Union[int, Any] = 5,
        dtype = int,
        weight = "bold",
        fontsize:Union[int, float]=12,
        max_xtick_val=None,
        min_xtick_val=None,
        **kwargs
):
    """

    :param ax:
    :param max_ticks:
        maximum number of ticks, if not set, all the default ticks will be used
    :param dtype:
    :param weight:
    :param fontsize:
    :param max_xtick_val:
        maxikum value of tick
    :param min_xtick_val:
    :return:
    """
    return set_ticklabels(ax, "x", max_ticks, dtype, weight, fontsize,
                          max_tick_val=max_xtick_val,
                          min_tick_val=min_xtick_val,
                          **kwargs)


def set_yticklabels(
        ax:plt.Axes,
        max_ticks:Union[int, Any] = 5,
        dtype=int,
        weight="bold",
        fontsize:int=12,
        max_ytick_val = None,
        min_ytick_val = None,
        **kwargs
):
    return set_ticklabels(
        ax, "y", max_ticks, dtype, weight,
        fontsize=fontsize,
        max_tick_val=max_ytick_val,
        min_tick_val=min_ytick_val,
        **kwargs
    )


def set_ticklabels(
        ax:plt.Axes,
        which:str = "x",
        max_ticks:int = 5,
        dtype=int,
        weight="bold",
        fontsize:int=12,
        max_tick_val = None,
        min_tick_val = None,
        **kwargs
):
    """

    :param ax:
    :param which:
    :param max_ticks:
    :param dtype:
    :param weight:
    :param fontsize:
    :param max_tick_val:
    :param min_tick_val:
    :param kwargs:
        any keyword arguments of axes.set_{x/y}ticklabels()
    :return:
    """
    ticks_ = getattr(ax, f"get_{which}ticks")()
    ticks = np.array(ticks_)
    if len(ticks)<1:
        warnings.warn(f"can not get {which}ticks {ticks_}")
        return

    if max_ticks:
        ticks = np.linspace(min_tick_val or min(ticks), max_tick_val or max(ticks), max_ticks)

    ticks = ticks.astype(dtype)

    getattr(ax, f"set_{which}ticks")(ticks)

    getattr(ax, f"set_{which}ticklabels")(ticks,
                                          weight=weight,
                                          fontsize=fontsize,
                                          **kwargs
                                          )
    return ax

# %%

def set_rcParams(**kwargs):
    plt.rcParams.update({'axes.labelsize': '14'})
    plt.rcParams.update({'axes.labelweight': 'bold'})
    plt.rcParams.update({'xtick.labelsize': '12'})
    plt.rcParams.update({'ytick.labelsize': '12'})
    plt.rcParams.update({'font.weight': 'bold'})
    plt.rcParams.update({'legend.title_fontsize': '12'})

    if sys.platform == "linux":

        _kwargs['font.family'] = 'serif'
        _kwargs['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    else:
        _kwargs['font.family'] = "Times New Roman"

    for k,v in kwargs.items():
        plt.rcParams[k] = v
    return

# %%

def residual_plot(
        train_true,
        train_prediction,
        test_true,
        test_prediction,
        label="Prediction",
        show:bool = False
):
    fig, axis = plt.subplots(1, 2, sharey="all"
                             , gridspec_kw={'width_ratios': [2, 1]})
    test_y = test_true.reshape(-1, ) - test_prediction.reshape(-1, )
    train_y = train_true.reshape(-1, ) - train_prediction.reshape(-1, )
    train_hist_kws = dict(bins=20, linewidth=0.5,
                          edgecolor="k", grid=False, color="#009E73",
                          orientation='horizontal')
    hist(train_y, show=False, ax=axis[1],
         label="Training", **train_hist_kws)
    plot(train_prediction, train_y, 'o', show=False,
         ax=axis[0],
         color="#009E73",
         markerfacecolor="#009E73",
         markeredgecolor="black", markeredgewidth=0.5,
         alpha=0.7, label="Training"
         )
    _hist_kws = dict(bins=40, linewidth=0.5,
                     edgecolor="k", grid=False,
                     color=np.array([225, 121, 144]) / 256.0,
                     orientation='horizontal')
    hist(test_y, show=False, ax=axis[1],
         **_hist_kws)

    set_xticklabels(axis[1], 3)

    plot(test_prediction, test_y, 'o', show=False,
         ax=axis[0],
         color="darksalmon",
         markerfacecolor=np.array([225, 121, 144]) / 256.0,
         markeredgecolor="black", markeredgewidth=0.5,
         ax_kws=dict(
             xlabel=label,
             ylabel="Residual",
             legend_kws=dict(loc="upper left"),
         ),
         alpha=0.7, label="Test",
         )
    set_xticklabels(axis[0], 5)
    set_yticklabels(axis[0], 5)
    axis[0].axhline(0.0, color="black")
    plt.subplots_adjust(wspace=0.15)

    if show:
       plt.show()
    return

# %%

def regression_plot(
        train_true,
        train_pred,
        test_true,
        test_pred,
        label,
        max_xtick_val = None,
        max_ytick_val = None,
        min_xtick_val=None,
        min_ytick_val=None,
        max_ticks = 5,
        show=False
):
    TRAIN_RIDGE_LINE_KWS = [{'color': '#009E73', 'lw': 1.0},
                            {'color': '#009E73', 'lw': 1.0}]
    TRAIN_HIST_KWS = [{'color': '#009E73', 'bins': 50},
                      {'color': '#009E73', 'bins': 50}]

    ax = regplot(train_true, train_pred,
                 marker_size=35,
                 marker_color="#009E73",
                 line_color='k',
                 fill_color='k',
                 scatter_kws={'edgecolors': 'black',
                              'linewidth': 0.5,
                              'alpha': 0.5,
                              },
                 label="Training",
                 show=False
                 )

    axHistx, axHisty = AddMarginalPlots(
        ax,
        ridge=False,
        pad=0.25,
        size=0.7,
        ridge_line_kws=TRAIN_RIDGE_LINE_KWS,
        hist_kws=TRAIN_HIST_KWS
    )(train_true, train_pred)

    train_r2 = RegressionMetrics(train_true, train_pred).r2()
    test_r2 = RegressionMetrics(test_true, test_pred).r2()
    ax.annotate(f'Training $R^2$= {round(train_r2, 2)}',
                xy=(0.95, 0.30),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")
    ax.annotate(f'Test $R^2$= {round(test_r2, 2)}',
                xy=(0.95, 0.20),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")

    ax_ = regplot(test_true, test_pred,
                  marker_size=35,
                  marker_color=np.array([225, 121, 144]) / 256.0,
                  line_style=None,
                  scatter_kws={'edgecolors': 'black',
                               'linewidth': 0.5,
                               'alpha': 0.5,
                               },
                  show=False,
                  label="Test",
                  ax=ax
                  )

    ax_.legend(fontsize=12, prop=dict(weight="bold"))
    TEST_RIDGE_LINE_KWS = [{'color': np.array([225, 121, 144]) / 256.0, 'lw': 1.0},
                           {'color': np.array([225, 121, 144]) / 256.0, 'lw': 1.0}]
    TEST_HIST_KWS = [{'color': np.array([225, 121, 144]) / 256.0, 'bins': 50},
                     {'color': np.array([225, 121, 144]) / 256.0, 'bins': 50}]
    AddMarginalPlots(
        ax,
        ridge=False,
        pad=0.25,
        size=0.7,
        ridge_line_kws=TEST_RIDGE_LINE_KWS,
        hist_kws=TEST_HIST_KWS
    )(test_true, test_pred, axHistx, axHisty)

    set_xticklabels(
        ax_,
        max_xtick_val=max_xtick_val,
        min_xtick_val=min_xtick_val,
        max_ticks=max_ticks,
    )
    set_yticklabels(
        ax_,
        max_ytick_val=max_ytick_val,
        min_ytick_val=min_ytick_val,
        max_ticks=max_ticks
    )
    ax.set_xlabel(f"Observed {label}")
    ax.set_ylabel(f"Predicted {label}")

    if show:
        plt.show()
    return ax

# %%

def ci_from_dist(
        distribution,
        coverage:float,
        true_array:np.ndarray,
        label:str,
        fill_color,
        line_color,
):
    """
    plots confidence interval from distribution

    :param coverage:
    :param distribution:
    :param true_array:
    :param label:
    :param fill_color:
    :param line_color:
    :return:
    """
    lower_90, upper_90 = distribution.interval(coverage)
    axes = plot(true_array, show=False,
              color=line_color,
              label='True')

    axes.fill_between(np.arange(len(lower_90)),
                    lower_90,
                    upper_90,
                    color=fill_color,
                    label=f"{int(coverage*100)}% CI",
                    alpha=0.6
                    )
    plt.legend()
    xticks = np.array(axes.get_xticks()).astype(int)
    axes.set_xticklabels(xticks, weight="bold", fontsize=12)
    yticks = np.array(axes.get_yticks()).astype(int)
    axes.set_yticklabels(yticks, weight="bold", fontsize=12)
    axes.set_xlabel('Samples', weight="bold", fontsize=12)
    axes.set_ylabel(f"Total {label}", weight="bold", fontsize=12)
    axes.grid(visible=True, ls='--', color='lightgrey')
    return

# %%

def plot_1d_pdp(pdp, train_data, feature, show=True):
    """1D pdp"""
    pdp_vals, ice_vals = pdp.calc_pdp_1dim(train_data, feature)

    ax = pdp.plot_pdp_1dim(pdp_vals, ice_vals, train_data,
                            feature,
                            pdp_line_kws={'color': '#5f3946'},
                            ice_color="#c8c0aa"
                            )
    ax.set_xlabel(COLUMN_MAPS_.get(feature, feature))
    ax.set_ylabel(f"E[f(x) | " + feature + "]")
    if show:
        plt.tight_layout()
        plt.show()

    return

# %%

def plot_stds(
        mean:np.ndarray,
        std:np.ndarray,
        label:str,
        pediction:np.ndarray = None,
        num_stds:int = 3,
        show:bool = True
):
    """
    plots standard deviations around mean/prediction array

    """
    if pediction is None:
        ax = plot(mean, show=False, color="grey", label="$\mu$",
                  ax_kws=dict(ylabel=label, xlabel="Samples"),
                  )
    else:
        ax = plot(pediction, show=False, color="grey", label="pediction",
                  ax_kws=dict(ylabel=label, xlabel="Samples"),
                  )

    if num_stds >= 3:
        ax.fill_between(np.arange(len(std)),
                        mean - (3 * std),
                        mean + (3 * std),
                        color="lightsteelblue",
                        label="$\mu$ $\u00B1$ 3 $\sigma$",
                        )

    if num_stds >= 2:
        ax.fill_between(np.arange(len(std)),
                        mean - (2 * std),
                        mean + (2 * std),
                        color="cornflowerblue",
                        label="$\mu$ $\u00B1$ 2 $\sigma$"
                        )

    ax.fill_between(np.arange(len(std)),
                    mean - std,
                    mean + std,
                    color="royalblue",
                    label="$\mu$ $\u00B1$  $\sigma$"
                    )
    plt.legend()

    if show:
        plt.tight_layout()
        plt.show()
    return

# %%

def version_info()->dict:
    info = get_version_info()
    info['ngboost'] = ngboost.__version__
    info['shap'] = shap.__version__
    return info
    

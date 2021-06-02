import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats
import numpy as np
import statsmodels.api as sm
import statsmodels

def linear_fit(x, y):
    """Returns bias and slope from regression y on x."""
    x = np.array(x)
    y = np.array(y)

    covs = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, covs)
    result = model.fit()
    return result.params, result.rsquared

def format(plt, nomp=False, notex=False):
    SMALL_SIZE = 16
    MEDIUM_SIZE = 24

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    # plt.style.use('seaborn-paper')
    if notex:
        import streamlit as st
        mode = st.selectbox('Mode', ['default', 'dark_background', 'Solarize_Light2',
         '_classic_test_patch',
         'bmh',
         'classic',
         'fast',
         'fivethirtyeight',
         'ggplot',
         'grayscale',
         'seaborn',
         'seaborn-bright',
         'seaborn-colorblind',
         'seaborn-dark',
         'seaborn-dark-palette',
         'seaborn-darkgrid',
         'seaborn-deep',
         'seaborn-muted',
         'seaborn-notebook',
         'seaborn-paper',
         'seaborn-pastel',
         'seaborn-poster',
         'seaborn-talk',
         'seaborn-ticks',
         'seaborn-white',
         'seaborn-whitegrid',
         'tableau-colorblind10'])
        plt.style.use(mode)
    if not notex:
        plt.rc("text.latex", preamble=r"\usepackage{bbold}")
    if nomp:
        return
    from matplotlib import rc

    if not notex:
        rc("text", usetex=True)



def get_plot(xrange, yrange, scaling, figsize=(9,8), notex=False):

    format(plt, notex=notex)
    if scaling == 'probit':
        def h(p):
            return scipy.stats.norm.ppf(p)
    elif scaling == 'logit':
        def h(p):
            return np.log(p / (1-p))
    else:
        def h(p):
            return p

    def transform(z):
        return [h(p) for p in z]


    fig, axlist = plt.subplots(1, 1, figsize=figsize)
    ax = axlist

    tick_loc_x = [round(z, 2) for z in np.arange(xrange[0], xrange[1], 0.05)]
    ax.set_xticks(transform(tick_loc_x))
    ax.set_xticklabels([str(int(loc * 100)) for loc in tick_loc_x])

    tick_loc_y = [round(z, 2) for z in np.arange(yrange[0], yrange[1], 0.05)]
    ax.set_yticks(transform(tick_loc_y))
    ax.set_yticklabels([str(int(loc * 100)) for loc in tick_loc_y])

    z = np.arange(min(xrange[0], yrange[0]), max(xrange[1], yrange[1]), 0.01)
    ax.plot(transform(z), transform(z), color="#A8A8A8", ls='--', label='$y = x$')

    ax.set_ylim(h(yrange[0]),h(yrange[1]))
    ax.set_xlim(h(xrange[0]),h(xrange[1]))

    ax.grid()

    return fig, axlist, transform

    #
    #
    # testbed_df = pd.read_json('/Users/mitchnw/git/test_clip/results_dbs/testbed_fmow.jsonl', lines=True).dropna()
    # clip_df = pd.read_json('/Users/mitchnw/git/open_clip/src/evaluation/misc/results_dbs/fmow.jsonl', lines=True).dropna()
    # ensemble_df = clip_df[clip_df['method'] == 'wilds_zeroshot_regression_ensemble']
    # interpolate_df = clip_df[clip_df['method'] == 'wilds_zeroshot_regression_interpolate']
    # l2_ablation_df = pd.read_json('/Users/mitchnw/git/open_clip/src/evaluation/misc/results_dbs/fmow_l2_ablation.jsonl', lines=True).dropna()
    #
    #
    # testbed_df = testbed_df[testbed_df[y] > 0.05]
    # testbed_df = testbed_df[testbed_df[x] > 0.05]
    #
    # transform_xrange = transform(xrange)
    # plt_xs = np.arange(transform_xrange[0], transform_xrange[1], 0.01)
    # linfit_params, _ = linear_fit(transform(testbed_df[x]), transform(testbed_df[y]))
    # ax.plot(plt_xs, plt_xs * linfit_params[1] + linfit_params[0], linewidth=2)
    #
    #
    # groups = testbed_df.groupby('model_name')
    # for i, (name, group) in enumerate(groups):
    #     tx = transform(group[x])
    #     ty = transform(group[y])
    #     ax.scatter(
    #         tx,
    #         ty,
    #         label=name,
    #         alpha=0.5,
    #         color=f'C{i}'
    #     )
    #
    #     # get ebs
    #     n = 10000
    #     x_cis = statsmodels.stats.proportion.proportion_confint(group[x] * 10000, 10000, alpha=0.05, method='beta')
    #     x_cis = (transform(x_cis[1]), transform(x_cis[0]))
    #     y_cis = statsmodels.stats.proportion.proportion_confint(group[y] * 10000, 10000, alpha=0.05, method='beta')
    #     y_cis = (transform(y_cis[0]), transform(y_cis[1]))
    #
    #     # ax.errorbar(transform(group[x]), transform(group[y]),
    #     #             yerr=y_cis, xerr=x_cis, linestyle="None",
    #     #             color=f'C{i}')
    #     # print('here')
    #
    # groups = clip_df[clip_df['alpha'] == 0].groupby('subtype')
    # for name, group in groups:
    #     print(name)
    #     ax.scatter(
    #         transform(group[clip_x]),
    #         transform(group[clip_y]),
    #         label=f"CLIP Zero Shot ({name})",
    #         alpha=0.66,
    #         s=400,
    #         marker='*',
    #     )
    #
    # groups = clip_df[clip_df['alpha'] == 1].groupby('subtype')
    # for name, group in groups:
    #     if 'ViT' in name:
    #         ax.scatter(
    #             transform(group[clip_x]),
    #             transform(group[clip_y]),
    #             label=f"CLIP Fine Tune ({name}) ($\lambda = 0.316$, $T = 1000$)",
    #             alpha=0.66,
    #             s=300,
    #             marker='d',
    #         )
    #
    # for name, group in l2_ablation_df.groupby('iters'):
    #     if 1000 == name:
    #         out = ax.scatter(
    #             transform(group[clip_x]),
    #             transform(group[clip_y]),
    #             label=f"CLIP Fine Tune (ViT-B/32), ($T={name}$)",
    #             alpha=0.66,
    #             s=100,
    #             marker='s',
    #             c=np.log10(group['lam']),
    #             cmap='magma'
    #         )
    #         cbar = fig.colorbar(out)
    #         cbar.set_label(r'$\textrm{log}_{10}(\lambda)$')
    #
    #
    # ax.set_xlabel('in-distribution accuracy', fontsize=18)
    # if 'worst' in y:
    #     ax.set_ylabel('out-of-distribution worst region accuracy', fontsize=18)
    # else:
    #     ax.set_ylabel('out-of-distribution accuracy', fontsize=18)
    # ax.set_title(f'Effect of Regression L2 Regularizer $\\lambda$. fMoW with scaling = {scaling}')
    # ax.grid()
    #
    #
    # fig.subplots_adjust(
    #     top=0.97, left=0.07, right=0.9, bottom=0.3, wspace=0.15, hspace=0.23
    # )
    # legend = ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor = (0.5, -0.14),
    #     #bbox_to_anchor=(0.425, -0.25),
    #     ncol=2,
    #     fontsize=18,
    # )
    #
    # plt.show()
    # # plt.savefig('fmow_l2_ablation_v2.pdf',
    # #             bbox_inches='tight')

    # # get ebs
    # n = 10000
    # x_cis = statsmodels.stats.proportion.proportion_confint(group[x] * 10000, 10000, alpha=0.05, method='beta')
    # x_cis = (transform(x_cis[1]), transform(x_cis[0]))
    # y_cis = statsmodels.stats.proportion.proportion_confint(group[y] * 10000, 10000, alpha=0.05, method='beta')
    # y_cis = (transform(y_cis[0]), transform(y_cis[1]))
    #
    # # ax.errorbar(transform(group[x]), transform(group[y]),
    # #             yerr=y_cis, xerr=x_cis, linestyle="None",
    # #             color=f'C{i}')
    # # print('here')


def add_linear_trend(ax, xs, ys, xrange, transform, color='C3', linewidth=2):
    transform_xrange = transform(list(xrange))
    plt_xs = np.arange(transform_xrange[0], transform_xrange[1], 0.01)
    linfit_params, _ = linear_fit(transform(xs), transform(ys))
    ax.plot(plt_xs, plt_xs * linfit_params[1] + linfit_params[0], linewidth=linewidth, color=color)


#
# if __name__ == '__main__':
#
#     xrange = (.05, .65)
#     yrange = (.05, .50)
#     scaling = 'probit'
#     fig, axlist, transform = get_plot(xrange, yrange, scaling)
#     ax = axlist
#
#     # Add fmow testbed points.
#     # TODO change path.
#     testbed_df = pd.read_json('/Users/mitchnw/git/test_clip/results_dbs/testbed_fmow.jsonl', lines=True).dropna()
#     x = 'FMoW-id_test:acc_avg'
#     y = 'FMoW-ood_test:acc_worst_region'
#     ax.set_xlabel(x.replace('_', ' '), fontsize=20)
#     ax.set_ylabel(y.replace('_', ' '), fontsize=20)
#     testbed_df = testbed_df[testbed_df[x] > 0.05]
#     testbed_df = testbed_df[testbed_df[y] > 0.05]
#     groups = testbed_df.groupby('model_name')
#     for i, (name, group) in enumerate(groups):
#         ax.scatter(
#             transform(group[x]),
#             transform(group[y]),
#             label=name,
#             alpha=0.5,
#             color=f'C{i}'
#         )
#     add_linear_trend(ax, testbed_df[x], testbed_df[y], xrange, transform)
#
#
#     # Add CLIP Zero Shot.
#     x = 'FMOWID:acc_avg'
#     y = 'FMOWOOD:acc_worst_region'
#     clip_df = pd.read_json(
#         '/Users/mitchnw/git/open_clip/src/evaluation/misc/results_dbs/fmow.jsonl',
#         lines=True
#     ).dropna()
#     groups = clip_df[clip_df['alpha'] == 0].groupby('subtype')
#     for name, group in groups:
#         print(name)
#         ax.scatter(
#             transform(group[x]),
#             transform(group[y]),
#             label=f"CLIP Zero Shot ({name})",
#             alpha=0.66,
#             s=400,
#             marker='*',
#         )
#
#     # Add the L2 poings.
#
#     # add legend.
#     fig.subplots_adjust(
#         top=0.97, left=0.07, right=0.9, bottom=0.3, wspace=0.15, hspace=0.23
#     )
#     legend = axlist.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.14),
#         ncol=2,
#         fontsize=18,
#     )
#
#     plt.show()
#     # plt.savefig('fmow_l2_ablation_v2.pdf',
#     #             bbox_inches='tight')
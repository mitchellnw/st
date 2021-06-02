import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats
import numpy as np
from utils import get_plot, add_linear_trend


hascb = False

colorinfo = {
    'RN50': 'C1',
    'RN101': 'C2',
    'RN50x4': 'C3',
    'ViT-B/32': 'C6'
}

def get_df(dataset):
    if 'CIFAR' in dataset:
        df = pd.read_json('results_dbs/cifar10.jsonl', lines=True).dropna()
    elif 'ImageNet' in dataset:
        df = pd.read_json('results_dbs/imagenet.jsonl', lines=True).dropna()
    elif 'FMOW' in dataset:
        df = pd.read_json('results_dbs/fmow.jsonl', lines=True).dropna()
    return df

def add_baselines(dataset, test_dataset, x=None, y=None):
    if 'ImageNet' in dataset:
        tx = x
        ty = y
        testbed_df = pd.read_json('results_dbs/testbed_imagenet.jsonl', lines=True).dropna(subset=[tx,ty])

    elif dataset == 'CIFAR10' and test_dataset == 'CIFAR10.2':
        tx = 'cifar10-test:top1'
        ty = 'cifar10.2-test:top1'
        testbed_df = pd.read_json('results_dbs/testbed_cifar10.jsonl', lines=True).dropna(subset=[tx,ty])

    elif dataset == 'FMOW':
        tx = f"FMoW-id_test:{':'.join(x.split(':')[1:])}"
        ty = f"FMoW-ood_test:{':'.join(y.split(':')[1:])}"
        testbed_df = pd.read_json('results_dbs/testbed_fmow.jsonl', lines=True).dropna(subset=[tx,ty])
        testbed_df = testbed_df[testbed_df[tx] > 0.05]
        testbed_df = testbed_df[testbed_df[ty] > 0.05]
    else:
        return

    ax.scatter(
        transform(testbed_df[tx]),
        transform(testbed_df[ty]),
        label='Baselines',
        alpha=0.5,
        color=f'C0'
    )
    add_linear_trend(ax, testbed_df[tx], testbed_df[ty], xrange, transform)

def add_baselines2(dataset, test_dataset):
    if dataset == 'CIFAR10' and test_dataset == 'CIFAR10.2':
        tx = 'cifar10-test:top1'
        ty = 'cifar10.2-test:top1'
        testbed_df = pd.read_json('results_dbs/testbed_cifar102.jsonl', lines=True).dropna(subset=[tx, ty])
        ax.scatter(
            transform(testbed_df[tx]),
            transform(testbed_df[ty]),
            label='Baselines (trained on CIFAR10.2)',
            alpha=0.5,
            color=f'C1'
        )
        add_linear_trend(ax, testbed_df[tx], testbed_df[ty], xrange, transform, color='C1', linewidth=1)

def add_zeroshot(model, dataset):
    df = get_df(dataset)
    if model != 'all':
        df = df[df['subtype'] == model]
    if 'CIFAR' in dataset or 'ImageNet' in dataset:
        df[x] = 0.01 * df[x]
        df[y] = 0.01 * df[y]
    zeroshot = df[df['method'] == 'zeroshot']
    for name, group in zeroshot.groupby('subtype'):
        if model != 'all' and model not in name:
            continue
        ax.scatter(
            transform(group[x]),
            transform(group[y]),
            label=f'Zeroshot ({name})',
            marker='*',
            s=400,
            alpha=0.5,
            c=colorinfo[name]
        )


def add_distill(model, dataset):
    df = get_df(dataset)

    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]

    interpolate = df[df['method'] == 'distill']
    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'Distill ({model})',
        marker='x',
        s=80,
        # c='C7',
        alpha=0.5,
        cmap='gist_earth',
        c=interpolate['alpha'],
    )
    global hascb
    if not hascb:
        hascb = True
        cbar = fig.colorbar(out)
        cbar.set_label(r'$\alpha$', fontsize=16)

def add_interpolate(interpolateion_type, model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = interpolateion_type.replace('/', '_').lower() + '_interpolate'
    marker = 'v' if method.startswith('random') else "+"
    sz = 30 if method.startswith('random') else 80

    interpolate = df[df['method'] == method]
    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'{interpolateion_type} (Interpolation, {model})',
        marker=marker,
        s=sz,
        alpha=0.5,
        cmap='gist_earth',
        c=interpolate['alpha'],
    )
    global hascb
    if not hascb:
        hascb = True
        cbar = fig.colorbar(out)
        cbar.set_label(r'$\alpha$', fontsize=16)

def add_line(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = 'random_probe_interpolate'
    df = df[df['method'] == method]

    e1 = df[df['alpha'] == 1.0]
    e2 = df[df['alpha'] == 0.0]

    x1 = e1[x]
    x2 = e2[x]

    y1 = e1[y]
    y2 = e2[y]
    if len(y2) != 1:
        print('error with add_line')
        return

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    alphas = np.arange(0, 1 + 0.01, 0.01)

    line_xs = (1 - alphas) * x1 + alphas * x2
    line_ys = (1 - alphas) * y1 + alphas * y2
    ax.plot(transform(line_xs), transform(line_ys), alpha = 0.5, c='C4', label='Random/Probe Line')

def add_transform_line(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = 'random_probe_interpolate'
    df = df[df['method'] == method]

    e1 = df[df['alpha'] == 1.0]
    e2 = df[df['alpha'] == 0.0]

    x1 = e1[x]
    x2 = e2[x]

    y1 = e1[y]
    y2 = e2[y]
    if len(y2) != 1:
        print('error with add_line')
        return

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    t = transform([x1, y1, x2, y2])
    x1, y1, x2, y2 = t[0], t[1], t[2], t[3]

    alphas = np.arange(0, 1 + 0.01, 0.01)

    line_xs = (1 - alphas) * x1 + alphas * x2
    line_ys = (1 - alphas) * y1 + alphas * y2
    ax.plot(line_xs, line_ys, alpha = 0.5, c='C1', label='Random/Probe Transform Line')

def add_ensemble(interpolateion_type, model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = interpolateion_type.replace('/', '_').lower() + '_ensemble'
    marker = '2' if method.startswith('random') else "^"
    interpolate = df[df['method'] == method]


    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'{interpolateion_type} (Ensemble, {model})',
        marker=marker,
        s=80,
        alpha=0.5,
        cmap='gist_earth',
        c=interpolate['alpha'],
    )
    global hascb
    if not hascb:
        hascb = True
        cbar = fig.colorbar(out)
        cbar.set_label(r'$\alpha$', fontsize=16)



def add_probe(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['method'] == 'zeroshot_probe_interpolate']
    df = df[df['alpha'] == 1.0]
    df = df[df['train_dataset'] == dataset]


    ax.scatter(
        transform(df[x]),
        transform(df[y]),
        label=f'Probe ({model})',
        marker='s',
        s=100,
        alpha=0.75,
        c=colorinfo[model]
    )

def add_probe_all(model, dataset):
    df = get_df(dataset)
    df = df[df['method'] == 'zeroshot_probe_interpolate']
    df = df[df['alpha'] == 1.0]
    df = df[df['train_dataset'] == dataset]


    for name, group in df.groupby('subtype'):
        ax.scatter(
            transform(group[x]),
            transform(group[y]),
            label=f'Probe ({name})',
            marker='s',
            s=100,
            alpha=0.75,
            c=colorinfo[name]
        )

def add_probe_vary_params(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['method'] == 'pytorch_probe']
    df = df[df['train_dataset'] == dataset]

    ax.scatter(
        transform(df[x]),
        transform(df[y]),
        label=f'Probe-Vary-Hyperparams ({model})',
        marker='1',
        s=100,
        alpha=0.5,
        c=colorinfo[model]
    )

def add_probe_vary_trainset_and_params(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['method'] == 'pytorch_probe']

    ax.scatter(
        transform(df[x]),
        transform(df[y]),
        label=f'Probe-Vary-Trainset-And-Hyperparams ({model})',
        marker='1',
        s=100,
        alpha=0.5,
        c=colorinfo[model]
    )


def add_probe_vary_trainset(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['method'] == 'zeroshot_probe_interpolate']
    df = df[df['alpha'] == 1.0]


    ax.scatter(
        transform(df[x]),
        transform(df[y]),
        label=f'Probe-Vary-TrainSet ({model})',
        marker='p',
        s=100,
        alpha=0.5,
        c=colorinfo[model]
    )

def add_probe_warmstart(model, dataset):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['method'] == 'pytorch_warmstart_probe']
    df = df[df['train_dataset'] == dataset]

    ax.scatter(
        transform(df[x]),
        transform(df[y]),
        label=f'Probe-Warmstart ({model})',
        marker='h',
        s=60,
        alpha=0.5,
        c=colorinfo[model]
    )

if __name__ == '__main__':

    dataset = st.selectbox(
    'Trainset',
    ['ImageNet', 'ImageNet50', 'ImageNet25', 'CIFAR10', 'FMOW'])

    if dataset == 'CIFAR10':
        yaxis = st.selectbox(
            'yaxis',
            ['CIFAR10.2', 'CIFAR10.1'])

        xrange = st.slider("xrange", 0.001, 0.99, (0.5, 0.97), 0.01)
        yrange = st.slider("yrange", 0.001, 0.99, (0.4, 0.9), 0.01)
    elif dataset == 'FMOW':
        xaxis = st.selectbox(
            'xaxis',
            ['FMOWID:acc_avg',
             'FMOWID:acc_worst_region',
             "FMOWID:acc_region:Asia",
             "FMOWID:acc_region:Europe",
             "FMOWID:acc_region:Africa",
             "FMOWID:acc_region:Americas",
             "FMOWID:acc_region:Oceania",
             "FMOWID:acc_region:Other",
             ])
        yaxis = st.selectbox(
            'yaxis',
            ['FMOWOOD:acc_worst_region',
             'FMOWOOD:acc_avg',
             "FMOWOOD:acc_region:Asia",
             "FMOWOOD:acc_region:Europe",
             "FMOWOOD:acc_region:Africa",
             "FMOWOOD:acc_region:Americas",
             "FMOWOOD:acc_region:Oceania",
             "FMOWOOD:acc_region:Other",
             ])
        xrange = st.slider("xrange", 0.001, 0.99, (0.05, 0.6), 0.01)
        yrange = st.slider("yrange", 0.001, 0.99, (0.05, 0.6), 0.01)
    else:
        yaxis = None
        xrange = st.slider("xrange", 0.001, 0.99, (0.5, 0.8), 0.01)
        yrange = st.slider("yrange", 0.001, 0.99, (0.4, 0.7), 0.01)

    scaling = st.selectbox(
    'Scaling',
    ['logit', 'probit', 'none'])


    model = st.selectbox(
    'Model',
    ['RN50x4', 'RN50', 'RN101', 'ViT-B/32'])


    options_list = ['Baselines', 'Zeroshot (all)',
     'Zeroshot',
     'Distill',
     'Interpolate (Zeroshot/Probe)',
     'Ensemble (Zeroshot/Probe)',
     'Probe',
     'Probe (all)',
     'Probe-Vary-Hyperparams',
     'Probe-Vary-TrainSet',
     'Probe-Warmstart',
     'Probe-Vary-TrainSet-And-Hyperparams',
     'Interpolate (Random/Probe)',
     'Ensemble (Random/Probe)',
     ]

    if dataset == 'CIFAR10':
        options_list.append('Baselines (trained on CIFAR10.2)')

    options = st.multiselect(
    'Options',
    options_list,
    ['Baselines', 'Zeroshot', 'Probe'])

    experimental_options = []


    fig, axlist, transform = get_plot(xrange, yrange, scaling, figsize=(10,8), notex=True)
    ax = axlist
    ax.set_title(f'scaling = {scaling}', fontsize=16)


    if 'ImageNet' in dataset:
        x = 'ImageNet:top1'
        y = 'ImageNetV2:top1'
    elif dataset == 'CIFAR10':
        x = 'CIFAR10:top1'
        y = 'CIFAR101:top1' if '.1' in yaxis else 'CIFAR102:top1'
    elif dataset == 'FMOW':
        x = xaxis#'FMOWID:acc_avg'
        y = yaxis#'FMOWOOD:acc_avg'


    ax.set_xlabel(x.replace('_', ' ').replace(':top1', ' '), fontsize=16)
    ax.set_ylabel(y.replace('_', ' ').replace(':top1', ' '), fontsize=16)


    if 'Baselines' in options:
        add_baselines(dataset, yaxis, x, y)

    if 'Baselines (trained on CIFAR10.2)' in options:
        add_baselines2(dataset, yaxis)

    if 'Zeroshot (all)' in options:
        add_zeroshot('all', dataset)
    elif 'Zeroshot' in options:
        add_zeroshot(model, dataset)

    if 'Probe (all)' in options:
        add_probe_all(model, dataset)
    elif 'Probe' in options:
        add_probe(model, dataset)

    if 'Distill' in options:
        add_distill(model, dataset)

    if 'Interpolate (Zeroshot/Probe)' in options:
        add_interpolate('Zeroshot/Probe', model, dataset)

    if 'Ensemble (Zeroshot/Probe)' in options:
        add_ensemble('Zeroshot/Probe', model, dataset)


    if 'Probe-Vary-TrainSet-And-Hyperparams' in options:
        add_probe_vary_trainset_and_params(model, dataset)
    else:
        if 'Probe-Vary-Hyperparams' in options:
            add_probe_vary_params(model, dataset)

        if 'Probe-Vary-TrainSet' in options:
            add_probe_vary_trainset(model, dataset)



    if 'Probe-Warmstart' in options:
        add_probe_warmstart(model, dataset)

    if 'Interpolate (Random/Probe)' in options:
        add_interpolate('Random/Probe', model, dataset)

    if 'Ensemble (Random/Probe)' in options:
        add_ensemble('Random/Probe', model, dataset)

    if 'Line (Random/Probe)' in experimental_options:
        add_line(model, dataset)
    if 'Transform Line (Random/Probe)' in experimental_options:
        add_transform_line(model, dataset)

    # add legend.
    fig.subplots_adjust(
        top=0.97, left=0.07, right=0.9, bottom=0.3, wspace=0.15, hspace=0.23
    )
    legend = axlist.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=16,
    )

    st.pyplot(fig)
    #plt.show()
    #plt.savefig('plots/imagenet_probe.pdf', bbox_inches='tight')


    experimental_options = st.multiselect(
    'Experimental Options',
    [
     'Line (Random/Probe)',
     'Transform Line (Random/Probe)',
     ],
    [])

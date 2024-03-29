import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats
import numpy as np
from utils import get_plot, add_linear_trend
from state import provide_state
import json

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
    elif 'IWildCam' in dataset:
        df = pd.read_json('results_dbs/iwildcam.jsonl', lines=True).dropna()
    elif 'Cars' in dataset:
        df = pd.read_json('results_dbs/cars.jsonl', lines=True).dropna()
    elif 'Pets' in dataset:
        df = pd.read_json('results_dbs/pets.jsonl', lines=True).dropna()
    elif 'Food101' in dataset:
        df = pd.read_json('results_dbs/food101.jsonl', lines=True).dropna()
    elif 'Caltech101' in dataset:
        df = pd.read_json('results_dbs/caltech101.jsonl', lines=True).dropna()
    return df

def add_baselines(dataset, test_dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    if 'ImageNet' in dataset:
        tx = x
        ty = y
        testbed_df = pd.read_json('results_dbs/testbed_imagenet.jsonl', lines=True).dropna(subset=[tx,ty])
        testbed_df[tx] = testbed_df[tx] * 0.01
        testbed_df[ty] = testbed_df[ty] * 0.01
        cmap = {
            'Standard training': 'tab:blue',
            'Lp adversarially robust': 'tab:olive',
           'Other robustness intervention': 'tab:brown',
           'Trained with more data': 'tab:green',
        }
        for name, group in testbed_df.groupby('type'):
            ax.scatter(
                transform(group[tx]),
                transform(group[ty]),
                label=name,
                alpha=0.5,
                color=cmap[name]
            )
        add_linear_trend(ax, testbed_df[tx], testbed_df[ty], xrange, transform)
        return
    elif dataset == 'CIFAR10':
        if test_dataset == 'CIFAR10.2':
            tx = 'cifar10_accuracy'
            ty = 'cifar10.2-test_accuracy'
            testbed_df = pd.read_csv('results_dbs/johns_cifar102.csv').dropna(subset=[tx,ty])
        elif test_dataset == 'CIFAR10.1':
            tx = 'cifar10_accuracy'
            ty = 'cifar10.1-v6_accuracy'
            testbed_df = pd.read_csv('results_dbs/johns_cifar101.csv').dropna(subset=[tx,ty])

        imgn_pretrained = testbed_df[testbed_df['pretrained']]
        trained =  testbed_df[~testbed_df['pretrained']]
        ax.scatter(
            transform(trained[tx]),
            transform(trained[ty]),
            label='Baseline',
            alpha=0.5,
            color='C0'
        )
        ax.scatter(
            transform(imgn_pretrained[tx]),
            transform(imgn_pretrained[ty]),
            label='Baseline (pretrained)',
            alpha=0.5,
            color='tab:green'
        )

        add_linear_trend(ax, testbed_df[tx], testbed_df[ty], xrange, transform)
        return

    elif dataset == 'FMOW':
        tx = f"FMoW-id_test:{':'.join(x.split(':')[1:])}"
        ty = f"FMoW-ood_test:{':'.join(y.split(':')[1:])}"
        testbed_df = pd.read_json('results_dbs/testbed_fmow.jsonl', lines=True).dropna(subset=[tx,ty])
        testbed_df = testbed_df[testbed_df[tx] > 0.05]
        testbed_df = testbed_df[testbed_df[ty] > 0.05]

        imgn_pretrained = testbed_df[testbed_df['model_name'] == 'Imagenet Pretrained Neural Network']
        trained =  testbed_df[testbed_df['model_name'] != 'Imagenet Pretrained Neural Network']
        ax.scatter(
            transform(trained[tx]),
            transform(trained[ty]),
            label='Baseline',
            alpha=0.5,
            color='C0'
        )
        ax.scatter(
            transform(imgn_pretrained[tx]),
            transform(imgn_pretrained[ty]),
            label='Baseline (pretrained)',
            alpha=0.5,
            color='tab:green'
        )

        add_linear_trend(ax, testbed_df[tx], testbed_df[ty], xrange, transform)
        return
    elif dataset == 'IWildCam':
        tx = f"IWildCamOfficialV2-id_test:{':'.join(x.split(':')[1:])}"
        ty = f"IWildCamOfficialV2-ood_test:{':'.join(y.split(':')[1:])}"
        testbed_df = pd.read_json('results_dbs/testbed_iwildcam.jsonl', lines=True).dropna(subset=[tx,ty])
        testbed_df = testbed_df[testbed_df[tx] > 0.03]
        testbed_df = testbed_df[testbed_df[ty] > 0.03]


        imgn_pretrained = testbed_df[testbed_df['model_name'] == 'Imagenet Pretrained Neural Network']
        trained =  testbed_df[testbed_df['model_name'] != 'Imagenet Pretrained Neural Network']
        ax.scatter(
            transform(trained[tx]),
            transform(trained[ty]),
            label='Baseline',
            alpha=0.5,
            color='C0'
        )
        ax.scatter(
            transform(imgn_pretrained[tx]),
            transform(imgn_pretrained[ty]),
            label='Baseline (pretrained)',
            alpha=0.5,
            color='tab:green'
        )

        add_linear_trend(ax, testbed_df[tx], testbed_df[ty], xrange, transform)
        return

def add_baselines2(dataset, test_dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_zeroshot(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    if model != 'all':
        df = df[df['subtype'] == model]
    if 'CIFAR' in dataset or 'ImageNet' in dataset or dataset == 'Cars' or dataset == 'Pets' or dataset == 'Caltech101' or dataset == 'Food101':
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


def add_distill(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_interpolate(interpolateion_type, model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_line(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_transform_line(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_ensemble(interpolateion_type, model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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


def add_old_temp_ensemble(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = 'zeroshot_probe_temperature_ensemble'
    marker = '3'
    interpolate = df[df['method'] == method]


    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'Zeroshot/Probe (Old Temperature Ensemble, {model})',
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


def add_temp_ensemble(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = 'zeroshot_probe_temp_ensemble'
    marker = 'o'
    interpolate = df[df['method'] == method]


    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'Zeroshot/Probe (Temperature Ensemble, {model})',
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

def add_logit_ensemble(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = 'zeroshot_probe_logit_ensemble'
    marker = 'P'
    interpolate = df[df['method'] == method]


    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'Zeroshot/Probe (Logit Ensemble, {model})',
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

def add_softmax_ensemble(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[df['train_dataset'] == dataset]
    method = 'zeroshot_probe_softmax_ensemble'
    marker = 'd'
    interpolate = df[df['method'] == method]


    out = ax.scatter(
        transform(interpolate[x]),
        transform(interpolate[y]),
        label=f'Zeroshot/Probe (Softmax Ensemble, {model})',
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

def add_probe(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_probe_all(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_probe_vary_params(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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

def add_probe_vary_trainset_and_params(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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


def add_probe_vary_trainset(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
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


def add_probe_warmstart(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[ df['method'] == 'pytorch_warmstart_probe']
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

def add_probe_warmstart_v2(model, dataset, x=None, y=None, fig=None, ax=None, transform=None, xrange=None):
    df = get_df(dataset)
    df = df[df['subtype'] == model]
    df = df[ df['method'] == 'pytorch_warmstart_probe_v2']
    df = df[df['train_dataset'] == dataset]

    out = ax.scatter(
        transform(df[x]),
        transform(df[y]),
        label=f'Probe-Warmstart-Regularize-To-Init ({model})',
        marker='h',
        s=60,
        alpha=0.5,
        c=df['alpha']
    )
    global hascb
    if not hascb:
        hascb = True
        cbar = fig.colorbar(out)
        cbar.set_label(r'$\alpha$', fontsize=16)


# def add_probe_warmstart(model, dataset):
#     df = get_df(dataset)
#     df = df[df['subtype'] == model]
#     df = df[np.logical_or(df['method'] == 'pytorch_warmstart_probe', df['method'] == 'pytorch_warmstart_probe_v2')]
#     df = df[df['train_dataset'] == dataset]
#
#     ax.scatter(
#         transform(df[x]),
#         transform(df[y]),
#         label=f'Probe-Warmstart ({model})',
#         marker='h',
#         s=60,
#         alpha=0.5,
#         c=colorinfo[model]
#     )

#
# def main(state):
#     query_params = st.experimental_get_query_params()
#     if "conf" in query_params:
#         state.conf = json.loads(query_params["conf"][0])
#     state.conf = state.conf or INITIAL_CONF
#
#     state.conf["checkbox"] = st.checkbox("Retain State !", value=state.conf["checkbox"])
#     state.conf["number"] = st.number_input("You Too Retain State !", value=state.conf["number"])
#     st.experimental_set_query_params(**{"conf": json.dumps(state.conf)})
#
# main()

def put_first(item, item_list):
    if item in item_list:
        item_list.remove(item)
        return [item] + item_list
    return item_list

INITIAL_CONF = {
    "dataset": 'ImageNet',
    'xaxis' : 'ImageNet:top1',
    'yaxis' : 'ImageNetV2:top1',
    'xrange' : (0.4, 0.8),
    'yrange' : (0.3, 0.7),
    'model' : 'RN50x4',
    'scaling' : 'logit',
    'options' : ['Baselines', 'Zeroshot', 'Probe']
}

@provide_state
def main(state):
    # Create an empty slot for the main plot, which we fill later.
    plot_slot = st.empty()

    query_params = st.experimental_get_query_params()
    if "conf" in query_params:
        state.conf = json.loads(query_params["conf"][0])
    state.conf = state.conf or INITIAL_CONF

    dataset = st.selectbox(
        'Trainset',
        put_first(
            state.conf["dataset"],
            ['ImageNet', 'ImageNet50', 'ImageNet25', 'CIFAR10', 'FMOW', 'IWildCam', 'Cars', 'Pets', 'Food101', 'Caltech101']
        )
    )

    savexrange = True
    xrange = st.slider("xrange", 0.001, 0.99, (state.conf['xrange'][0], state.conf['xrange'][1]), 0.01)
    yrange = st.slider("yrange", 0.001, 0.99, (state.conf['yrange'][0], state.conf['yrange'][1]), 0.01)


    if 'ImageNet' in dataset:
        xaxis = st.selectbox(
            'xaxis',
            put_first(state.conf['xaxis'], ['ImageNet:top1',
             'ImageNetRValClasses:top1',
             'ImageNetAValClasses:top1',
             'ObjectNetValClasses:top1',
             ]))
        yaxis = st.selectbox(
            'yaxis',
            put_first(state.conf['yaxis'], ['ImageNetV2:top1',
             'ImageNetR:top1',
             'ImageNetA:top1',
             'ObjectNet:top1',
             'ImageNetSketch:top1',
             ]))
        # xrange = st.slider("xrange", 0.001, 0.99, (state.conf['xrange'][0], state.conf['xrange'][1]), 0.01)
        # yrange = st.slider("yrange", 0.001, 0.99, (state.conf['yrange'][0], state.conf['yrange'][1]), 0.01)

    elif dataset == 'CIFAR10':
        xaxis = None
        savexaxis = False
        yaxis = st.selectbox(
            'yaxis',
             put_first(state.conf['yaxis'], ['CIFAR10.2', 'CIFAR10.1']))
        #
        # xrange = st.slider("xrange", 0.001, 0.99, (0.5, 0.97), 0.01)
        # yrange = st.slider("yrange", 0.001, 0.99, (0.4, 0.9), 0.01)
    elif dataset == 'FMOW':
        xaxis = st.selectbox(
            'xaxis',
            put_first(state.conf['xaxis'],['FMOWID:acc_avg',
             'FMOWID:acc_worst_region',
             "FMOWID:acc_region:Asia",
             "FMOWID:acc_region:Europe",
             "FMOWID:acc_region:Africa",
             "FMOWID:acc_region:Americas",
             "FMOWID:acc_region:Oceania",
             "FMOWID:acc_region:Other",
             ]))
        yaxis = st.selectbox(
            'yaxis',
            put_first(state.conf['yaxis'], ['FMOWOOD:acc_worst_region',
             'FMOWOOD:acc_avg',
             "FMOWOOD:acc_region:Asia",
             "FMOWOOD:acc_region:Europe",
             "FMOWOOD:acc_region:Africa",
             "FMOWOOD:acc_region:Americas",
             "FMOWOOD:acc_region:Oceania",
             "FMOWOOD:acc_region:Other",
             ]))
        # xrange = st.slider("xrange", 0.001, 0.99, (0.05, 0.6), 0.01)
        # yrange = st.slider("yrange", 0.001, 0.99, (0.05, 0.6), 0.01)
    elif dataset == 'IWildCam':
        xaxis = st.selectbox(
            'xaxis',
            put_first(state.conf['xaxis'],  ['IWildCamID:acc_avg',
             'IWildCamID:F1-macro_all',
             ]))
        yaxis = st.selectbox(
            'yaxis',
            put_first(state.conf['yaxis'], ['IWildCamOOD:acc_avg',
             'IWildCamOOD:F1-macro_all',
             ]))
        # xrange = st.slider("xrange", 0.001, 0.99, (0.05, 0.6), 0.01)
        # yrange = st.slider("yrange", 0.001, 0.99, (0.05, 0.6), 0.01)
    elif dataset == 'Cars' or dataset == 'Pets' or dataset == 'Caltech101' or dataset == 'Food101':
        xaxis = 'alpha'
        yaxis = f'{dataset}:top1'
        xrange = (-0.2, 1.2)
        savexrange=False
        # yrange = st.slider("yrange", 0.001, 0.99, (0.001, 0.99), 0.01)

    else:
        yaxis = None
        # xrange = st.slider("xrange", 0.001, 0.99, (0.5, 0.8), 0.01)
        # yrange = st.slider("yrange", 0.001, 0.99, (0.4, 0.7), 0.01)



    skip_linear = False
    if dataset == 'Cars' or dataset == 'Pets' or dataset == 'Caltech101' or dataset == 'Food101':
        scaling = 'none'
        skip_linear = True
        scaling_options = ['none']
    else:
        scaling_options = put_first(state.conf['scaling'], ['logit', 'probit', 'none'])
    scaling = st.selectbox(
    'Scaling',
    scaling_options
    )


    model = st.selectbox(
    'Model',
    put_first(state.conf['model'], ['RN50x4', 'RN50', 'RN101', 'ViT-B/32']))


    options_list = ['Baselines', 'Zeroshot (all)',
     'Zeroshot',
     'Distill',
     'Interpolate (Zeroshot/Probe)',
     'Ensemble (Zeroshot/Probe)',
     'Temperature-Ensemble (Zeroshot/Probe)',
     'Logit-Ensemble (Zeroshot/Probe)',
     'Softmax-Ensemble (Zeroshot/Probe)',
     'Probe',
     'Probe (all)',
     'Probe-Vary-Hyperparams',
     'Probe-Vary-TrainSet',
     'Probe-Warmstart',
     'Probe-Warmstart-Regularize-To-Init',
     'Probe-Vary-TrainSet-And-Hyperparams',
     'Interpolate (Random/Probe)',
     'Ensemble (Random/Probe)',
     'Old-Temperature-Ensemble (Zeroshot/Probe)',
     ]

    if dataset == 'CIFAR10':
        options_list.append('Baselines (trained on CIFAR10.2)')
    state_options = state.conf['options']

    options = st.multiselect(
    'Options',
    options_list,
    state_options)

    experimental_options = []

    fig, axlist, transform = get_plot(xrange, yrange, scaling, figsize=(10,8), notex=True, skip_linear=skip_linear)
    ax = axlist
    ax.set_title(f'scaling = {scaling}', fontsize=16)


    if dataset == 'CIFAR10':
        x = 'CIFAR10:top1'
        y = 'CIFAR101:top1' if '.1' in yaxis else 'CIFAR102:top1'
    else:
        x = xaxis#'FMOWID:acc_avg'
        y = yaxis#'FMOWOOD:acc_avg'


    ax.set_xlabel(x.replace('_', ' ').replace(':top1', ' '), fontsize=16)
    ax.set_ylabel(y.replace('_', ' ').replace(':top1', ' '), fontsize=16)


    if 'Baselines' in options:
        add_baselines(dataset, yaxis, x, y, fig, ax, transform, xrange)

    if 'Baselines (trained on CIFAR10.2)' in options:
        add_baselines2(dataset, yaxis, x, y, fig, ax, transform, xrange)

    if 'Zeroshot (all)' in options:
        add_zeroshot('all', dataset, x, y, fig, ax, transform, xrange)
    elif 'Zeroshot' in options:
        add_zeroshot(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Probe (all)' in options:
        add_probe_all(model, dataset, x, y, fig, ax, transform, xrange)
    elif 'Probe' in options:
        add_probe(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Distill' in options:
        add_distill(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Interpolate (Zeroshot/Probe)' in options:
        add_interpolate('Zeroshot/Probe', model, dataset, x, y, fig, ax, transform, xrange)

    if 'Ensemble (Zeroshot/Probe)' in options:
        add_ensemble('Zeroshot/Probe', model, dataset, x, y, fig, ax, transform, xrange)



    if 'Temperature-Ensemble (Zeroshot/Probe)' in options:
        add_temp_ensemble(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Logit-Ensemble (Zeroshot/Probe)' in options:
        add_logit_ensemble(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Softmax-Ensemble (Zeroshot/Probe)' in options:
        add_softmax_ensemble(model, dataset, x, y, fig, ax, transform, xrange)


    if 'Probe-Vary-TrainSet-And-Hyperparams' in options:
        add_probe_vary_trainset_and_params(model, dataset, x, y, fig, ax, transform, xrange)
    else:
        if 'Probe-Vary-Hyperparams' in options:
            add_probe_vary_params(model, dataset, x, y, fig, ax, transform, xrange)

        if 'Probe-Vary-TrainSet' in options:
            add_probe_vary_trainset(model, dataset, x, y, fig, ax, transform, xrange)



    if 'Probe-Warmstart' in options:
        add_probe_warmstart(model, dataset, x, y, fig, ax, transform, xrange)
    if 'Probe-Warmstart-Regularize-To-Init' in options:
        add_probe_warmstart_v2(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Interpolate (Random/Probe)' in options:
        add_interpolate('Random/Probe', model, dataset, x, y, fig, ax, transform, xrange)

    if 'Ensemble (Random/Probe)' in options:
        add_ensemble('Random/Probe', model, dataset, x, y, fig, ax, transform, xrange)


    if 'Old-Temperature-Ensemble (Zeroshot/Probe)' in options:
        add_old_temp_ensemble(model, dataset, x, y, fig, ax, transform, xrange)

    if 'Line (Random/Probe)' in experimental_options:
        add_line(model, dataset, x, y, fig, ax, transform, xrange)
    if 'Transform Line (Random/Probe)' in experimental_options:
        add_transform_line(model, dataset, x, y, fig, ax, transform, xrange)

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

    plot_slot.pyplot(fig)
    #plt.show()
    #plt.savefig('plots/imagenet_probe.pdf', bbox_inches='tight')


    experimental_options = st.multiselect(
    'Experimental Options',
    [
     'Line (Random/Probe)',
     'Transform Line (Random/Probe)',
     ],
    [])


    state.conf["dataset"] = dataset
    state.conf["xaxis"] = xaxis
    state.conf["yaxis"] = yaxis
    if savexrange: state.conf["xrange"] = xrange
    state.conf["yrange"] = yrange
    state.conf["scaling"] = scaling
    state.conf["model"] = model
    state.conf["options"] = options
    st.experimental_set_query_params(**{"conf": json.dumps(state.conf)})

    text = (
        "Description (use horizontal scroll if necessary):\n\n"
        'Zeroshot (all) : Zeroshot accuracy for all models in the CLIP repo.\n\n'
         'Zeroshot : Zeroshot accuracy for the selected model.\n\n'
         'Distill : Train a probe with objective (1-alpha)\*loss(pred, soft_zeroshot_targets) + alpha \* loss(pred, supervised_targets).\n\n'
         'Interpolate (Zeroshot/Probe) : With probability 1-alpha use the zeroshot classifier else use the probe.\n\n'
         'Ensemble (Zeroshot/Probe) : Deprecated. Use Temperature/Logit/Softmax ensemble below.\n\n'
         'Temperature-Ensemble (Zeroshot/Probe) : Make predictions (alpha \* 1000 \* zeroshot_logits).softmax() + probe_logits.softmax() \n\n'
         'Logit-Ensemble (Zeroshot/Probe) : Make predictions (1-alpha) \* zeroshot_logits + alpha \* probe_logits.\n\n'
         'Softmax-Ensemble (Zeroshot/Probe) : Make predictions (1-alpha) \* zeroshot_logits.softmax() + alpha \* probe_logits.softmax().\n\n'
         'Probe : Accuracy of the probe trained on the visual features.\n\n'
         'Probe (all) : Accuracy of the probe trained on the visual features for all models made available by OpenAI.\n\n'
         'Probe-Vary-Hyperparams : Vary the hyperparameters (weight decay/learning rate/epochs) when training the linear probe.\n\n'
         'Probe-Vary-TrainSet : Only available for ImageNet. Vary the number of examples used to train the probe.\n\n'
         'Probe-Warmstart : (May have bugs) start the probe training with the zeroshot classifier.\n\n'
         'Probe-Warmstart-Regularize-To-Init : (May have bugs) start the probe training with the zeroshot classifier and regularize to this initialization.\n\n'
         'Probe-Vary-TrainSet-And-Hyperparams : Combines Probe-Vary-Hyperparams and Probe-Vary-Trainset.\n\n'
         'Interpolate (Random/Probe) : With probability 1-alpha use a random classifier else use the probe.\n\n'
         'Ensemble (Random/Probe) : Make predictions (1-alpha) \* random_logits + alpha \* probe_logits.\n\n'
         'Old-Temperature-Ensemble : (Zeroshot/Probe) Deprecated. Use Temperature/Logit/Softmax ensemble above.\n\n'
    )
    for t in text.split("\n\n"):
        st.write(t)

    st.write(state.conf)

if __name__ == '__main__':
    st.write("Refresh or clear url if things get weird. Slack for bugs / requests.")
    eg = "http://localhost:8502/?conf=%7B%22dataset%22%3A+%22Cars%22%2C+%22xaxis%22%3A+%22alpha%22%2C+%22yaxis%22%3A+%22Cars%3Atop1%22%2C+%22xrange%22%3A+%5B-0.2%2C+1.2%5D%2C+%22yrange%22%3A+%5B0.3%2C+0.94%5D%2C+%22model%22%3A+%22RN50x4%22%2C+%22scaling%22%3A+%22none%22%2C+%22options%22%3A+%5B%22Baselines%22%2C+%22Zeroshot%22%2C+%22Probe%22%2C+%22Interpolate+%28Zeroshot%2FProbe%29%22%5D%7D"
    main()
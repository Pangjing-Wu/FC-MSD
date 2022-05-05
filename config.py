import os


seed = 0
data_offset = 60 # max tau.


save_dir = './results'
path = dict(
    test_raw=os.path.join(save_dir, 'ms-clf', 'prediction', 'raw'),
    test_metrics=os.path.join(save_dir, 'ms-clf', 'prediction', 'metrics'),
    style=os.path.join(save_dir, 'ms-clf', 'market-style'),
    baseline_raw=os.path.join(save_dir, 'baseline', 'raw'),
    baseline_metrics=os.path.join(save_dir, 'baseline', 'metrics'),
    msa_gs=os.path.join(save_dir, 'grid-search', 'msa', 'results'),
    msa_param=os.path.join(save_dir, 'grid-search', 'msa', 'param'),
    clf_gs=os.path.join(save_dir, 'grid-search', 'classifier', 'results'),
    clf_param=os.path.join(save_dir, 'grid-search', 'classifier', 'param')
)
for p in path.values():
    os.makedirs(p, exist_ok=True)


class ConfigWrapper(object):
    
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


data = ConfigWrapper(
    data_dir='./dataset/processed',
    split=0.8,
    y_col='pct_chg',
    y_offset=1,
    norm=True,
    binarize=True
    )


cluster = ConfigWrapper(
    kmeans=dict(
        n_init=10
    ),
    hierarchy=dict(
        affinity='euclidean',
        linkage='ward',
        n_neighbors=1
    )
)


msa = ConfigWrapper(
    msa=dict(
        y_col='pct_chg',
        offset=data_offset
    ),
    grid_search=dict(
        n_clusters=list(range(2, 11)),
        taus=list(range(20, data_offset + 1, 5)),
    ),
    search_repeat=10
)


classifier = ConfigWrapper(
    svc=dict(
        kernel='rbf'
    ),
    gbdt=dict(
        max_features='sqrt',
        n_iter_no_change=5,
        validation_fraction=0.2
    ),
    adaboost=dict(
        algorithm='SAMME.R'
    ),
    grid_search=dict(
        n_jobs=10,
        cv=10
    ),
    grids = ConfigWrapper(
        gbdt=dict(
            learning_rate=[1e-4, 1e-3, 1e-2, 1e-1],
            n_estimators=[10, 20, 50, 100, 200],
            max_depth=[2, 3, 4, 5]
        ),
        adaboost=dict(
            learning_rate=[1e-4, 1e-3, 1e-2, 1e-1],
            n_estimators=[10, 20, 50, 100, 200]
        ),
        svc=dict(
            C=[1e6, 1e5, 1e4, 1e3, 1e2],
            gamma=['scale', 'auto', 1e-3, 1e-2, 1e-1]
        )
    )
)


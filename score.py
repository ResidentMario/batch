import numpy as np
import argparse
from joblib import load

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, dest='filename', help='path to the dataset to be scored')
args = parser.parse_args()

if __name__ == "__main__":
    from distributed import Client, LocalCluster
    from dask_ml.wrappers import ParallelPostFit
    import dask.dataframe as dd

    cluster = LocalCluster()
    client = Client(cluster)

    clf = load('wta-matches-model.joblib')
    clf = ParallelPostFit(clf)

    matches = dd.read_csv(args.filename, assume_missing=True)
    point_diff = (matches.winner_rank_points - matches.loser_rank_points).dropna()
    X_test = point_diff.compute().values[:, np.newaxis]

    y_test_pred = clf.predict(X_test)
    np.save("predictions.npy", y_test_pred)

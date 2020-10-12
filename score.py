import pandas as pd
import numpy as np
import argparse
from joblib import load

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, dest='filename', help='path to the dataset to be scored')
args = parser.parse_args()

clf = load('/mnt/model/wta-matches-model.joblib')

matches = pd.read_csv(args.filename)
point_diff = (matches.winner_rank_points - matches.loser_rank_points).dropna()
X_test = point_diff.values[:, np.newaxis]

y_test_pred = clf.predict(X_test)
np.save('pred.npy', y_test_pred)
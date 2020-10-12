import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump

matches = pd.read_csv("/mnt/wta-matches/wta_matches_2015.csv")
point_diff = (matches.winner_rank_points - matches.loser_rank_points).dropna()
X = point_diff.values[:, np.newaxis]
y = (point_diff > 0).values.astype(int).reshape(-1, 1)

sort_order = np.argsort(X[:, 0])
X = X[sort_order, :]
y = y[sort_order, :]

clf = LogisticRegression()
clf.fit(X, y.ravel())

dump(clf, '/spell/wta-matches-model.joblib') 
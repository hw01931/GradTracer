import xgboost as xgb
import json
import numpy as np

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
r = xgb.XGBClassifier(n_estimators=1, max_depth=2).fit(X, y)
d = r.get_booster().get_dump(dump_format='json')[0]
print(d)

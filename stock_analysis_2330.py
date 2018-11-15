# coding=utf-8
# Feature Importance
"""
 1) close                          0.317028
 2) high                           0.240169
 3) low                            0.231113
 4) open                           0.210046
 5) deal_count                     0.000495
 6) diff                           0.000478
 7) money                          0.000340
 8) volume                         0.000332
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("2330_v2.csv", header=0, sep='\t',
                    lineterminator='\r', encoding = 'utf-8')
data = data.dropna()
data_num = data.shape[0]

#data[:,8] = data[:,8].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
#remove comma
data.iloc[:,1] = data.iloc[:,1].str.replace(',','')
data.iloc[:,2] = data.iloc[:,2].str.replace(',','')
data.iloc[:,8] = data.iloc[:,8].str.replace(',','')

#convert datatype:object to numeric type
data = data.convert_objects(convert_numeric=True)

#generate label data: use the previous date's close price
data['label'] = data['close']

for i in range(0, data_num):
    if i < data_num -1:
        data.loc[i, 'label'] = data.loc[i+1, 'close']
    else:
        data.loc[i, 'label'] = data.loc[i, 'close']

# transform pandas to numpy
#data=data.iloc[:,1:10].values  #取第2-10列
trX = data.iloc[:,1:9]
trY = data.iloc[:,9]
feat_labels = data.columns[1:-1]

# Calculate moving average
from pandas.stats.moments import rolling_mean

date_range = data["date"].values
plt.plot(date_range, data["label"].values, label="close original")
plt.plot(date_range, rolling_mean(data, 5)["label"].values, label="close 5")
plt.plot(date_range, rolling_mean(data, 10)["label"].values, label="close 10")
plt.legend()
plt.show()
plt.gcf().clear()

#from sklearn.model_selection import train_test_split
#trX, teX, trY, teY = train_test_split(
#    X, y, test_size=0.2, random_state=0)

# Assessing Feature Importances with Random Forests
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)
forest.fit(trX, trY)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(trX.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(trX.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(trX.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, trX.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()


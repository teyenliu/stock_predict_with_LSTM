from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('2330_v2.csv', sep='\t', header=0, encoding = 'utf8')
data['volume'] = data['volume'].str.replace(',', '')
data['money'] = data['money'].str.replace(',', '')
data['deal_count'] = data['deal_count'].str.replace(',', '')
data['volume'] = pd.to_numeric(data['volume'])
data['money'] = pd.to_numeric(data['money'])
data['deal_count'] = pd.to_numeric(data['deal_count'])

# generate label data
data["label"] = data["close"]
for i in range(0, data["label"].count()-1):
    data["label"][i] = data["close"][i+1] - data["close"][i]
data.drop(data.index[data["label"].count() - 1])


data = data.dropna()
dataX = data.iloc[:,1:9].values  #取第2-9列
feat_labels = data.columns[1:9]
dataY = data.iloc[:,9].values  #取第10列

# Scale 
#scaler = preprocessing.MinMaxScaler()
#scalerX = scaler.fit(dataX)
#dataX = scalerX.transform(dataX)


# Split trining data and testing data
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
trX, teX, trY, teY = train_test_split(
    dataX, dataY, test_size=0.2, random_state=0)

# Assessing Feature Importances with Random Forests
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000,
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

trY_pred = forest.predict(trX)
teY_pred = forest.predict(teX)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(trY, trY_pred),
        mean_squared_error(teY, teY_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(trY, trY_pred),
        r2_score(teY, teY_pred)))

# Plot the chart
plt.figure()
plt.plot(teY[:], 'bo', teY_pred[:], 'k')
plt.legend(['Stock 2330 validation'], loc='upper left')
plt.show()

# Validation
dataX2 = np.array([[45492993, 10977665819, 244.00, 244.00, 239.00, 239.50, -3.00, 10071],[76396174,18582475956,244.50,244.50,242.00,242.50,+6.00,21512]])
dataY2 = np.array([245.0-239.50, 239.50-242.50])

#scalerX.transform(dataX2)
valY2_pred = forest.predict(dataX2)
print "forest forecast ==>", dataY2, ":", valY2_pred

print teY, ":", teY_pred


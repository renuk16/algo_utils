import pandas as pd

train = pd.read_csv("../input/train.csv", nrows=1000) 
test = pd.read_csv("../input/test.csv", nrows=1000)
print("Done reading data..")

train.target.head()

xgboost = XGboost(target='target', task="classification")

features = xgboost.keepcols(train)
model = xgboost.train(train, nt=10, val=True)
print("Model training done..")

test['target'] = xgboost.test(test)
test[['ID_code', 'target']].to_csv("submissions.csv", index=False)

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class XGboost():
    """
    Class for preprocessing, training, and scoring an xgboost model
    """

    def __init__(self, target, task):
        self.features = []
        self.target = target
        if task == "classification":
            self.task = "classification"
            self.model = XGBClassifier()
        elif task == "regression":
            self.task = "regression"
            self.model = XGBClassifier()
        else:
            print("task needs to be clasification or regression")

    def keepcols(self, data):
        
        for i in data.columns:
            if (data[i].dtype != 'object'):
                self.features.append(i)
        
        self.features.remove(self.target)

        return self.features
    
    def test(self, data, score = None):
        data_X = data[self.features]
        data_X.fillna(0, inplace=True)
        if score != None:
            data_y = data[self.target]
        
        pred = self.model.predict(data_X)
        
        if score == "rmse":
            return np.sqrt(((pred - data_y) ** 2).mean())
        elif score == "auc":
            return roc_auc_score(data_y, pred)
        
        if score == None:
            return pred

    def train(self, data, nt=1000,lr=0.01,depth=5,seed=None, verbose=1, val=False):
        
        if val == True:
            data, vald = train_test_split(data, test_size = 0.3)
        
        data_X = data[self.features]
        data_X.fillna(0, inplace=True)
        data_y = data[self.target]
        # train_y.fillna(1, inplace=True)

        # Lets start with a linear regression model. The model can be easily change to a
        # random forest or gbm
        if self.task == "regression":
            self.model = XGBRegressor(n_estimators=nt, learning_rate=lr, max_depth=depth, seed=seed, verbose=verbose)
            self.model.fit(data_X, data_y)
            
        elif self.task == "classification":
            self.model = XGBClassifier(n_estimators=nt, learning_rate=lr, max_depth=depth, seed=seed, verbose=verbose)
            self.model.fit(data_X, data_y)
            
        if val == True:
            print("The accuracy is {}".format(self.test(vald, "auc")))
        
        return self.model
    

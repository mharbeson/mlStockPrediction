from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

class Learning_Model:
    def __init__(self, predictors, training_data):
        self.model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2000,
            random_state=1
        )
        self.train = training_data.iloc[:-100]
        self.test = training_data.iloc[-100:]

        self.model.fit(self.train[predictors], self.train["Target"])

        self.predictors = self.model.predict(self.test[predictors])
        self.predictors = pd.Series()
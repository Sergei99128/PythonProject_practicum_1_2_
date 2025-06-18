# with open('data/parkinsons.data','r+', encoding='UTF-8') as file:
#     file.read()
#     print(file)
import numpy as np
import xgboost
from matplotlib import cm, pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# from xgboost import XGBClassifier
# # read data
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# data = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# # create model instance
# bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# # fit model
# bst.fit(X_train, y_train)
# # make predictions
# preds = bst.predict(X_test)


# from xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

class Disease:

    def __init__(self, track):
        # self.data = pd.DataFrame(np.arange(12).reshape(4 ,3 ), columns= ['a', 'b', 'c'])
        # self.data = xgboost.DMatrix('train.csv?format=csv&label_column=0')
        self.pf = pd.read_csv(track)
        # self.data['result'] = self.data[['name']]
        # self.data['status_disable'] = self.data['status'].map({1 : 'Parkinson', 0 : 'healthy'})
        # self.pf[''] =
    def open_file(self):
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
        bst.fit(X_train, y_train)
        # make predictions
        preds = bst.predict(X_test)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])
        # disp.plot(cmap='Blues')
        # plt.title("Confusion Matrix")
        # plt.show()
        print(f"Accuracy: {accuracy_score(X_test):.2%}")
        return preds
# pf = pd.read_csv('data/parkinsons.data')

track = 'data/parkinsons.data'
if __name__ == '__main__':
    disease = Disease(track)
    print(disease)
    # print(disease.open_file())
    # disease.open_file()
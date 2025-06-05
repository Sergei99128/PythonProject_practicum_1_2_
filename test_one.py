from keras.src.backend.jax.numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from time import  time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier


class News:

    def __init__(self,file):

        self.file = file

        self.pf = pd.read_csv(self.file)

    def content_news(self):

        self.pf['content'] = self.pf['title']+ " " +  self.pf['text']
        self.pf['label_num'] = self.pf['label'].map({'FAKE': 0 , 'REAL': 1})
        self.x = self.pf['content']
        self.y = self.pf['label_num']

    def train(self):

        x_train, x_test, y_train, y_test = train_test_split(
                                    self.x, self.y,
                                            test_size=0.2, random_state=46)

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        x_train_vect = vectorizer.transform(x_train)
        x_test_vect = vectorizer.transform(x_test)

        model = PassiveAggressiveClassifier(max_iter=2000)
        model.fit(x_train_vect,y_train)

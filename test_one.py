# from keras.src.backend.jax.numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from time import  time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class News:

    def __init__(self,file):

        self.file = file

        self.pf = pd.read_csv(self.file)

    def content_news(self):

        self.pf['content'] = self.pf['title']+ " " +  self.pf['text']
        self.pf['label_num'] = self.pf['label'].map({'FAKE': 0 , 'REAL': 1})
        self.x = self.pf['content']
        self.y = self.pf['label_num']
        return self.pf['content']
    def train(self):

        x_train, x_test, y_train, y_test = train_test_split(
                                    self.x, self.y,
                                            test_size=0.2, random_state=46)

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        x_train_vect = vectorizer.fit_transform(x_train)
        x_test_vect = vectorizer.fit_transform(x_test)

        model = PassiveAggressiveClassifier(max_iter=1000)
        model.fit(x_train_vect,y_train)

        y_pred = model.predict(x_test_vect)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()


if __name__ == '__main__':
    trac = 'data/fake_news.csv'
    news = News(trac)
    print(news.content_news())
    news.train()

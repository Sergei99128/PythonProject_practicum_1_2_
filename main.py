# Задача 1. Обнаружение фальшивых новостей
# Фальшивые новости — это ложная информация, распространяемая через социальные
# сети и другие сетевые СМИ для достижения политических или идеологических целей.
#
# Твоя  задача -  используя библиотеку sklearn построить модель классического
# машинного обучения, которая может с высокой точностью более 90% определять,
# является ли новость реальной (REAL） или фальшивой（FAKE).
#
# Ты должен самостоятельно изучить и применить к задаче TfidfVectorizer для извлечения
# признаков из текстовых данных и PassiveAggressiveClassifier.
#
# Ты  можешь использовать данный датасет для обучения.
#
# Построй матрицу ошибок (confusion matrix). Представь, что ваш заказчик
# очень любит графики и диаграммы. Визуализируй для него результаты там, где это возможно.
#
# Что такое TfidfVectorizer?
# Изучение текстовых данных является одной из фундаментальных задач в
# области анализа данных и машинного обучения.
#
# Однако тексты представляют собой сложные и многомерные структуры,
# которые не могут быть напрямую обработаны алгоритмами машинного обучения.
# В этом контексте извлечение признаков — это процесс преобразования текстовых
# данных в числовые векторы, которые могут быть использованы для обучения моделей и
# анализа. Этот шаг играет ключевую роль в предварительной обработке данных перед
# применением алгоритмов.
#
# Term Frequency-Inverse Document Frequency (TF-IDF) — это один из наиболее
# распространенных и мощных методов для извлечения признаков из текстовых данных.
# TF-IDF вычисляет важность каждого слова в документе относительно количества его
# употреблений в данном документе и во всей коллекции текстов. Этот метод позволяет
# выделить ключевые слова и понять, какие слова имеют больший вес для определенного
# документа в контексте всей коллекции.
#
# TfidfVectorizer преобразует коллекцию необработанных документов в матрицу объектов TF-IDF.
#
# Что такое пассивно-агрессивный классификатор (PassiveAggressiveClassifier)?
# Пассивно-агрессивный классификатор – это алгоритм онлайн-обучения, в котором вы обучаете
# систему постепенно, загружая ее экземпляры последовательно, отдельно или небольшими
# группами, называемыми мини-партиями.
#
# При онлайн-обучении модель машинного обучения обучается и развертывается в
# производственной среде таким образом, чтобы обучение продолжалось по мере поступления
# новых наборов данных. Таким образом, мы можем сказать, что такой алгоритм, как
# пассивно-агрессивный классификатор, лучше всего подходит для систем, которые получают
# данные в непрерывном потоке. Он пассивно реагирует на правильные классификации и агрессивно
# реагирует на любые просчеты.

import learn
import tensorflow as tf
from   sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from time import time

from sklearn.datasets import fetch_20newsgroups


class News:

    def __init__(self, trek):
        self.data = trek

        pf = pd.read_csv(self.data)
        self.pf = pf

    # def t(self):
    #     return self.y
    # def re(self):

    def size_mb(self,doc):
        return sum(len(s.encode("utf-8")) for s in self.pf ) / 1e6

    def load_dataset(self,verbose=False, remove=()):
        """Load and vectorize the 20 newsgroups dataset."""

    #     data_train = fetch_20newsgroups(
    #     subset="train",
    #     categories=categories,
    #     shuffle=True,
    #     random_state=42,
    #     remove=remove,
    # )
        data_train = self.pf

    #     data_test = fetch_20newsgroups(
    #     subset="test",
    #     categories=categories,
    #     shuffle=True,
    #     random_state=42,
    #     remove=remove,
    # )
        data_test = self.pf
        # order of labels in `target_names` can be different from `categories`
        # target_names = data_train.target_names

        # split target in a training set and a test set
        y_train, y_test = data_train.target, data_test.target

        # Extracting features from the training data using a sparse vectorizer
        t0 = time()
        vectorizer = TfidfVectorizer(
            sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
        )
        X_train = vectorizer.fit_transform(data_train.data)
        duration_train = time() - t0

        # Extracting features from the test data using the same vectorizer
        t0 = time()
        X_test = vectorizer.transform(data_test.data)
        duration_test = time() - t0

        feature_names = vectorizer.get_feature_names_out()

        if verbose:
            # compute size of loaded data
            data_train_size_mb = self.size_mb(data_train.data)
            data_test_size_mb = self.size_mb(data_test.data)

            print(
                f"{len(data_train.data)} documents - "
                f"{data_train_size_mb:.2f}MB (training set)"
            )
            print(f"{len(data_test.data)} documents - {data_test_size_mb:.2f}MB (test set)")
            print(f"{len(target_names)} categories")
            print(
                f"vectorize training done in {duration_train:.3f}s "
                f"at {data_train_size_mb / duration_train:.3f}MB/s"
            )
            print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
            print(
                f"vectorize testing done in {duration_test:.3f}s "
                f"at {data_test_size_mb / duration_test:.3f}MB/s"
            )
            print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

        return X_train, X_test, y_train, y_test, feature_names, target_names

if __name__ == '__main__':
    data = 'data/fake_news.csv'
    news = News(data)
    categories = ['FAKE']
    # news.size_mb(docs)
    news.load_dataset()
    # print(news.t())

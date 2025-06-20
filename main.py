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

import pandas as pd

df = pd.read_csv('data/fake_news.csv')

df['content'] = df['title'] + " " + df['text']

df['label_num'] = df['label'].map({'FAKE': 0, 'REAL': 1})

X = df['content']
y = df['label_num']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=10000)
model.fit(X_train_vec, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
# plt.show()
print(cm)
if __name__ == '__main__':
    print('is_ok')


# with open('data/parkinsons.data','r+', encoding='UTF-8') as file:
#     file.read()
#     print(file)
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
pf = pd.read_csv('data/parkinsons.data')
if __name__ == '__main__':
    print(pf)
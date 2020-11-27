from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import pandas as pd
import numpy as np

text_dict = {'text':[], 'german':[], 'french':[], 'spanish':[], 'english':[]}

def read_data(url):
    languages = ['german', 'french', 'spanish', 'english']
    train = pd.read_csv(url)
    for index, row in train.iterrows():
        sentence = row['text']
        language = row['language']
        text_dict['text'].append(sentence)
        for e in languages:
            if language == e:
                text_dict[language].append(1)
            else:
                text_dict[e].append(0)
    train = pd.DataFrame(text_dict)
    return(train)

train = read_data("https://raw.githubusercontent.com/ishantjuyal/Language-Detection/main/Dataset/languages.csv")
train.head()

Y = train[['german', 'french', 'spanish', 'english']]
Y = np.array(Y)
X = np.array(train['text'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_new = []
for line in X:
    token_list = tokenizer.texts_to_sequences([line])[0]
    X_new.append(token_list)
max_sequence_len = max(max([len(x) for x in X_new]), 100)
input_sequences = np.array(pad_sequences(X_new, maxlen=max_sequence_len, padding='pre'))
total_words = len(tokenizer.word_index) + 1
X = input_sequences

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Embedding(total_words, 32, input_length = X.shape[1]))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
print(model.summary())

history = model.fit(X_train, y_train, epochs = 1)

y_pred = np.argmax(model.predict(X_test), axis = -1)

y_test = [list(i).index(1) for i in list(y_test)]
y_test = np.array(y_test)

print("Accuracy of model on test set:")
print(accuracy_score(y_test, y_pred))
print("\nThe confusion matrix on test set")
print(confusion_matrix(y_test, y_pred))

def detect(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    zeros = max(0, max_sequence_len - len(token_list))
    token_list = [0]*zeros + token_list
    token_list = np.array(token_list).reshape(1, max_sequence_len)
    index = np.argmax(model.predict(token_list), axis = -1)[0]
    
    languages = ['german', 'french', 'spanish', 'english']
    return(languages[index])



import imageio as imageio
from keras import Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_model():
    # create model
    model = Sequential()
    model.add(Dense(768, input_dim=768, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(360, kernel_initializer='normal', activation='relu'))
    model.add(Dense(180, kernel_initializer='normal', activation='relu'))
    model.add(Dense(62, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def one_hot(c):
    list_of_zeros = [0] * 62
    if 48 <= ord(c) <= 57:
        list_of_zeros[ord(c) - 48] = 1
    if 65 <= ord(c) <= 90:
        list_of_zeros[ord(c) - 55] = 1
    if 97 <= ord(c) <= 122:
        list_of_zeros[ord(c) - 61] = 1
    return list_of_zeros


def load_data():
    x = list()
    y = list()
    for i in range(1, 6284):
        img = imageio.imread(f"train_greyscale/{i}.png")
        img.resize(16 * 16 * 3, )
        x.append(img)

    x_train = np.array(x)
    x_train = x_train / 255
    f = open('trainLabels.csv', 'r')

    reader = csv.reader(f)
    for row in reader:
        if row[1] == 'Class':
            continue
        y.append(one_hot(row[1]))
    y_train = np.array(y)

    return train_test_split(x_train, y_train, random_state=42, shuffle=True, test_size=0.2)


def main():
    X_train, X_test, Y_train, Y_test = load_data()

    model = build_model()
    history = model.fit(X_train, Y_train, batch_size=32, epochs=40)
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    score = model.evaluate(X_test, Y_test)

    print(score)


if __name__ == '__main__':
    main()

import os
import shutil
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class GlassPredict:
    def __init__(self, ):
        self.data = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.input_shape = None

    @staticmethod
    def draw_plot(history, output="plot"):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'val'], loc='upper left')

        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'val'], loc='upper left')
        fig.tight_layout()
        plt.show()
        fig.savefig(output + ".jpg")

    @staticmethod
    def extract_files(directory):
        for file in os.listdir(directory):
            for face in os.listdir(f'{directory}/{file}'):
                mode = 0 if face.split('_')[3] == 'open' else 1
                shutil.copyfile(f'{directory}/{file}/{face}', f'dataset/{mode}/{face}')

    def pre_process(self):
        self.extract_files("faces_4")

        self.data = []
        self.labels = []
        for f in os.listdir('dataset/0'):
            img = keras.preprocessing.image.load_img(path='dataset/0/' + f, color_mode='grayscale')
            self.data.append(np.array(img))
            self.labels.append(0)
        for f in os.listdir('dataset/1'):
            img = keras.preprocessing.image.load_img(path='dataset/1/' + f, color_mode='grayscale')
            self.data.append(np.array(img))
            self.labels.append(1)

        self.data = np.array(self.data, dtype=np.float16) / 255.0
        self.labels = np.array(self.labels, dtype=np.float16)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, random_state=42,
                                                                                stratify=self.labels, test_size=0.2)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train,
                                                                                  random_state=42,
                                                                                  stratify=self.y_train,
                                                                                  test_size=0.13)

        self.input_shape = self.X_train[0].shape

    def part3(self):

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=self.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(units=150, activation='relu'),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])

        model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

        tf.random.set_seed(100)
        history = model.fit(x=self.X_train, y=self.y_train, batch_size=16, epochs=60,
                            validation_data=(self.X_valid, self.y_valid), verbose=0)
        self.draw_plot(history, "part3")

    def part4(self):
        hist_dict = {}
        for n_neurons in [2, 4, 10, 50]:
            model = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=self.input_shape),
                keras.layers.Flatten(),
                keras.layers.Dense(n_neurons, activation='relu'),
                keras.layers.Dense(1, activation=keras.activations.sigmoid)
            ])

            model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
            hist_dict[n_neurons] = model.fit(x=self.X_train, y=self.y_train, batch_size=16, epochs=60,
                                             validation_data=(self.X_valid, self.y_valid), verbose=0)
            self.draw_plot(hist_dict[n_neurons], f"part4-{n_neurons}neurons")

    def part5(self):
        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=self.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])

        model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

        model.fit(x=self.X_train, y=self.y_train, batch_size=16, epochs=60,
                  validation_data=(self.X_valid, self.y_valid), verbose=0)

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=self.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])

        model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

        history = model.fit(x=self.X_train, y=self.y_train, batch_size=16, epochs=60,
                            validation_data=(self.X_valid, self.y_valid), verbose=0)
        self.draw_plot(history, "part5-with-2-layer")

        model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=self.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])

        model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

        history = model.fit(x=self.X_train, y=self.y_train, batch_size=16, epochs=60,
                            validation_data=(self.X_valid, self.y_valid), verbose=0)
        self.draw_plot(history, "part5-with-2-layer")

    def execute(self):
        self.pre_process()

        self.part3()
        self.part4()
        self.part5()


if __name__ == "__main__":
    GlassPredict().execute()

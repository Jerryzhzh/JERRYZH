from tensorflow.keras import Model, layers

class Yanwing_detection(Model):
    def __init__(self):
        super(Yanwing_detection, self).__init__()
        self.conv1 = layers.Conv2D(6, (5, 5), activation='relu')
        self.pooling1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(8, (5, 5), activation='relu')
        self.pooling2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(10, (4, 4), activation='relu')
        self.pooling3 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(2, activation='sigmoid')

    def call(self, x):
        conv1_output = self.conv1(x)
        pooling1_output = self.pooling1(conv1_output)
        conv2_output = self.conv2(pooling1_output)
        pooling2_output = self.pooling2(conv2_output)
        conv3_output = self.conv3(pooling2_output)
        pooling3_output = self.pooling3(conv3_output)
        flatten_layer = self.flatten(pooling3_output)
        predict = self.d1(flatten_layer)
        return self.d2(predict)

if __name__ == '__main__':
    model = Yanwing_detection()
    model.summary()
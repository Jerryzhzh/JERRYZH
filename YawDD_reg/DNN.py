from tensorflow.keras import Model, layers

class DNN_model(Model):
    def __init__(self):
        super(DNN_model, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(1024, activation='relu')
        self.dense3 = layers.Dense(2, activation='sigmoid')

    def call(self, x):
        inputs = self.flatten(x)
        dense1_output = self.dense1(inputs)
        dense2_output = self.dense2(dense1_output)
        return self.dense3(dense2_output)
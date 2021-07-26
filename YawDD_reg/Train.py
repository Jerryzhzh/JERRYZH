import os.path

import tensorflow as tf
from load_data import DataProcess
from pre_Yanwing_model import Yanwing_detection
from DNN import DNN_model

train_set, test_set = DataProcess()
model = Yanwing_detection()

# model.build(input_shape=(None, 240, 320, 3))

# model.summary()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
train_pre = tf.keras.metrics.Precision(name='train_pre')
train_recall = tf.keras.metrics.Recall(name='train_recall')
train_auc = tf.keras.metrics.AUC(name='train_auc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
test_pre = tf.keras.metrics.Precision(name='test_pre')
test_recall = tf.keras.metrics.Recall(name='test_recall')
test_auc = tf.keras.metrics.AUC(name='test_auc')

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    predictions = tf.math.argmax(predictions, axis=1)
    train_accuracy(labels, predictions)
    train_pre(labels, predictions)
    train_recall(labels, predictions)
    train_auc(labels, predictions)


def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    predictions = tf.math.argmax(predictions, axis=1)
    test_accuracy(labels, predictions)
    test_pre(labels, predictions)
    test_recall(labels, predictions)
    test_auc(labels, predictions)

EPOCHS = 5
print('--------------training--------------------')
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_pre.reset_states()
    train_recall.reset_states()
    train_auc.reset_states()

    test_loss.reset_states()
    test_accuracy.reset_states()
    test_pre.reset_states()
    test_recall.reset_states()
    test_auc.reset_states()

    for images, labels in train_set:
        inputs = tf.cast(images, dtype=tf.float32) / 255.0
        train_step(inputs, labels)

    for test_images, test_labels in test_set:
        test_inputs = tf.cast(test_images, dtype=tf.float32) / 255.0
        test_step(test_inputs, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, Auc: {}, Test Loss: {}, Test Accuracy: {}, Test Precision: {}, Test Recall: {}, Test Auc: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          train_pre.result()*100,
                          train_recall.result()*100,
                          train_auc.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100,
                          test_pre.result()*100,
                          test_recall.result()*100,
                          test_auc.result()*100))
print('--------------training done!--------------------')
save_path = './YawDD_model/CNN'
print('saving model at '+ save_path)
# model.save(save_path)
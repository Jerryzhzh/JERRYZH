import os.path
import tensorflow as tf
import cv2


warning_text = 'CAREFUL!'
print('--------------loading model!--------------------')
load_path = './YawDD_model'
model = tf.keras.models.load_model(load_path)
'''
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer, loss)
model.summary()
'''
print('--------------loading done!--------------------')

print('--------------loading video!--------------------')
# video_dir = 'D:/dataset/YawDD dataset/Mirror/Female_mirror'
# save_dir = '../YawDD dataset/Result'
# file_name = '34-FemaleNoGlasses-Yawning.avi'
# video_dir = '../YawDD dataset/Dash/Female'
# save_dir = '../YawDD dataset/Result'
# file_name = '9-FemaleNoGlasses.avi'
video_dir = 'D:/dataset/YawDD dataset/Mirror/Male_mirror Avi Videos'
save_dir = '../YawDD dataset/Result'
file_name = '7-MaleGlasses-Yawning.avi'
video_path = video_dir + '//' + file_name
videoCapture = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
save_path = os.path.join(save_dir, file_name)
out = cv2.VideoWriter(save_path, fourcc, 30, (320, 240))
print('--------------loading done!--------------------')
def process_image(frame):
    x, y = frame.shape[0:2]
    frame = cv2.resize(frame, (int(y / 2), int(x / 2)))
    inputs = tf.cast(tf.expand_dims(frame, 0), dtype=tf.float32) / 255.0
    is_warming = tf.nn.softmax(model(inputs))
    if is_warming[0][1] >= 0.5:
        cv2.putText(frame, warning_text, (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 2)
    return frame

while True:
    success, frame = videoCapture.read()
    if not success:
        print('video is all read')
        break
    clean_image = process_image(frame)
    out.write(clean_image)
videoCapture.release()
out.release()
cv2.destroyAllWindows()
'''
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

train_set, test_set = DataProcess()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


test_loss.reset_states()
test_accuracy.reset_states()

for test_images, test_labels in test_set:
    test_inputs = tf.cast(test_images, dtype=tf.float32) / 255.0
    test_step(test_inputs, test_labels)

template = 'Test Loss: {}, Test Accuracy: {}'
print(template.format(test_loss.result(),
                      test_accuracy.result()*100))
'''
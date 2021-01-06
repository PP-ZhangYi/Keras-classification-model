import time, os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def model_predict(model_path, imgs_path, img_size, classes):
    model = load_model(model_path)
    count = 0
    for file in os.listdir(imgs_path):
        file_path = os.path.join(imgs_path, file)
        # start_time = time.time()
        img = load_img(file_path, target_size=(img_size, img_size))
        input_x = img_to_array(img) / 255.
        input_x = np.expand_dims(input_x, axis=0)
        predict = model.predict(input_x)
        result = np.argmax(predict, axis=1)
        # print(result)
        if result[0] == classes:
            count += 1
        # end_time = time.time()
        # print('use_time:', end_time - start_time)
    acc = count / len(os.listdir(imgs_path))
    print('acc: ', acc)


if __name__ == '__main__':
    model_path = r'D:\PY_scipty\Keras\checkpoint\weishts.02_loss0.0465_inception.hdf5'
    imgs_path = r'D:\PY_scipty\Keras\data\class\val\2'
    img_size = 299
    classes = 2
    model_predict(model_path, imgs_path, img_size, classes)

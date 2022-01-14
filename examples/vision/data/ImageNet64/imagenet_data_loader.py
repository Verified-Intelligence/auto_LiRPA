import os

import numpy as np
from PIL import Image


class DatasetDownsampledImageNet():
    def __init__(self):
        # self.data_path = data_path
        os.mkdir('train')
        os.mkdir('test')
        for i in range(1000):
            os.mkdir('train/' + str(i))
            os.mkdir('test/' + str(i))
            print(i)
        self.load_data('raw_data/Imagenet64_train_npz', count=0, fname='train/')
        self.load_data('raw_data/Imagenet64_val_npz', count=1e8, fname='test/')

    def load_data(self, data_path, img_size=64, count=0., fname=''):
        files = os.listdir(data_path)
        img_size2 = img_size * img_size

        # count = 0  # 1e8  # test data start with 1
        for file in files:
            f = np.load(data_path + '/' + file)
            x = np.array(f['data'])
            y = np.array(f['labels']) - 1
            x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3))

            for i, img in enumerate(x):
                img = Image.fromarray(img.reshape(img_size, img_size, 3))
                name = str(int(count)).zfill(9)
                label = str(y[i])
                print(count, fname + label + '/' + name + '_label_' + label.zfill(4) + '.png')
                # img.show()
                img.save(fname + label + '/' + name + '_label_' + label.zfill(4) + '.png')

                count += 1


if __name__ == "__main__":
    DatasetDownsampledImageNet()

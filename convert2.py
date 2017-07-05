import os
import cv2
import sys
import numpy as np

import cifar10


def convert_train():
    labels = {'Kashiwaghi_Yuki_':1, 'Komima_Haruna_':2, 'Sashihara_Rino_':3, 'Watanabe_Mayu_':4, 'Yamamoto_Sayaka_':5}
    name_list = ['Kashiwaghi_Yuki_', 'Yamamoto_Sayaka_', 'Komima_Haruna_', 'Sashihara_Rino_', 'Watanabe_Mayu_']
    data_size = [247, 121, 194, 98, 113]
    output_file = open('data/cifar10_data/cifar-10-batches-bin/train_batch.bin', 'ab')
    for k in range(5):
        for i in range(data_size[k]):
            name = name_list[k]
            label = labels[name]
            img_name='{0}{1:03}.jpg'.format(name, i)
            img_path = 'data/akb_train/' + img_name
            if os.path.isfile(img_path):
                print(img_path)
                im = cv2.imread(img_path)
                im = cv2.resize(im, (32, 32))

                r = im[:,:,0].flatten()
                g = im[:,:,1].flatten()
                b = im[:,:,2].flatten()
                output = np.array([label] + list(r) + list(g) + list(b), dtype = np.uint8)
                # output = np.array([label] + list(im.flatten()), dtype = np.uint8)
                output.tofile(output_file)
    output_file.close()

def convert_test():
    output_file = open('data/cifar10_data/cifar-10-batches-bin/test_batch.bin', 'ab')
    for i in range(250, 265):
        name = 'Kashiwaghi_Yuki_'
        label = 1
        img_name='{0}{1:03}.jpg'.format(name, i)
        img_path = 'data/akb_test/' + img_name
        if os.path.isfile(img_path):
            print(img_path)
            im = cv2.imread(img_path)
            im = cv2.resize(im, (32, 32))

            r = im[:,:,0].flatten()
            g = im[:,:,1].flatten()
            b = im[:,:,2].flatten()
            output = np.array([label] + list(r) + list(g) + list(b), dtype = np.uint8)
            # output = np.array([label] + list(im.flatten()), dtype = np.uint8)
            output.tofile(output_file)
    output_file.close()


if __name__ == '__main__':
    convert_train()
    convert_test()



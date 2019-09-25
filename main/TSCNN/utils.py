import json
import os
import numpy as np
from PIL import Image


def split_train_data(data_list, val_split):
    num_val = int(len(data_list) * val_split)
    num_train = len(data_list) - num_val
    return num_val, num_train


def make_teacher_data(base_path, teacher_directory_list, timesteps, split_data=0.1):
    train_data = []
    val_data = []
    for teacher_directory, counter in zip(teacher_directory_list, range(len(teacher_directory_list))):
        directory_path = base_path + teacher_directory
        directory_contain_list = os.listdir(directory_path)

        teacher_data = []
        for directory_contain in directory_contain_list:
            teacher_data_list_path = directory_path + '\\' + directory_contain
            teacher_data_list = os.listdir(teacher_data_list_path)

            for teacher_data_number in range(len(teacher_data_list) - timesteps):
                image_list = []
                axis_list = []
                for t in range(timesteps):
                    teacher_data_path = teacher_data_list_path + '\\' + teacher_data_list[teacher_data_number + t]
                    opt_image_path = teacher_data_path + '\\' + '00_crop_bgr_image.jpg'
                    rgb_image_path = teacher_data_path + '\\' + '00_crop_image.jpg'
                    axis_data_path = teacher_data_path + '\\' + '00_crop_image.json'

                    with open(axis_data_path, 'r') as f:
                        json_data = json.load(f)
                    x = int(json_data['x'])
                    y = int(json_data['y'])
                    w = int(json_data['w'])
                    h = int(json_data['h'])

                    label = [1 if i == counter else 0 for i in range(len(teacher_directory_list))]

                    image_list.append([opt_image_path, rgb_image_path])
                    axis_list.append([x, y, w, h])
                teacher_data.append([image_list, label, axis_list])

        num_val, num_train = split_train_data(teacher_data, split_data)
        train_data.extend(teacher_data[:num_train])
        val_data.extend(teacher_data[num_train:])
    return train_data, val_data


def scale_change(image, aspect_width, aspect_height):
    """resize image with unchanged aspect ratio using padding"""
    img_width = image.size[0]
    img_height = image.size[1]

    if img_width > img_height:
        rate = aspect_width / img_width
    else:
        rate = aspect_height / img_height
    image = image.resize((int(img_width * rate), int(img_height * rate)))
    iw, ih = image.size
    new_image = Image.new('RGB', (aspect_width, aspect_height), (128, 128, 128))
    new_image.paste(image, ((aspect_width - iw) // 2, (aspect_height - ih) // 2))
    return new_image


def image_open(image_path, width, height):
    image = Image.open(image_path)
    image = scale_change(image, width, height)
    image_array = np.asarray(image)
    return image_array


def data_generator(data_list, batch_size, time_sequence, image_shape=(224, 224)):
    '''data generator for fit_generator'''
    n = len(data_list)
    i = 0
    width = image_shape[0]
    height = image_shape[1]
    while True:

        axis = []
        label_list = []
        axis_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data_list)
            image_data_array = []
            batch_data = []
            for t in range(time_sequence):
                opt_image_array = image_open(data_list[i][0][t][0], width, height)
                original_image_array = image_open(data_list[i][0][t][1], width, height)

                image_array = np.append(opt_image_array, original_image_array, axis=2)
                image_data_array.append(image_array)

                seq_data = [data for data in data_list[i][2][t]]
                batch_data.append(seq_data)
            axis_data.append(batch_data)

            label = [l for l in data_list[i][1]]
            axis.append(image_data_array)
            label_list.append(label)
            i = (i + 1) % n

        image_data = np.array(axis)
        label = np.array(label_list)
        axis = np.array(axis_data)
        yield [image_data, axis], label

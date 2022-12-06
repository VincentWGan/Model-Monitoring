import numpy as np
import pandas as pd
import os
import cv2
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.utils import img_to_array
import random


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def imageDataStream(input_dir, batch_size):
    model2 = ResNet50(weights='imagenet')
    # Arrays needed to store the amount of correct, incorrect, and valid predictions
    num_valid, num_invalid, num_pred_wrong, num_weak = 0, 0, 0, 0

    # Loading in the image, and defining the random variable to determine which image should be selected
    for _ in range(batch_size):
        img = None
        while img is None:
            img_path = os.path.join(input_dir, random.choice(os.listdir(input_dir)))
            img = cv2.imread(img_path)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # Determining the prediction of the truth image
        z = img_to_array(img)
        z = np.expand_dims(z, axis=0)
        z = preprocess_input(z)
        preds_true = model2.predict(z)

        # First Manipulation--Increasing Brightness
        random_num_1 = random.randint(1, 250)
        bright_img = change_brightness(img, random_num_1)
        x = img_to_array(bright_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds_1 = model2.predict(x)

        # Second Manipulation--Decreasing Brightness
        random_num_2 = random.randint(1, 250) * (-1)
        bright_img_ = change_brightness(img, random_num_2)
        y = img_to_array(bright_img_)
        y = np.expand_dims(y, axis=0)
        y = preprocess_input(y)
        preds_2 = model2.predict(y)

        # Determines within each manipulated image the final determination--is this a cat or a dog
        a = decode_predictions(preds_1, top=3)
        list1, list2, list3 = zip(*a)
        a_class_1 = (list1[0][0])
        a_class_2 = (list2[0][0])
        a_class_3 = (list3[0][0])

        a = [*a_class_1]
        b = [*a_class_2]
        c = [*a_class_3]
        truth_trigger_brightened = False

        if a[2] in [b[2], c[2]]:
            truth_trigger_brightened = True

        b = decode_predictions(preds_2, top=3)
        list1, list2, list3 = zip(*b)
        b_class_1 = (list1[0][0])
        b_class_2 = (list2[0][0])
        b_class_3 = (list3[0][0])

        a = [*b_class_1]
        b = [*b_class_2]
        c = [*b_class_3]
        truth_trigger_dulled = False

        if a[2] in [b[2], c[2]]:
            truth_trigger_dulled = True

        # Is the internal prediction within the class correct--Is the dog a dog, is the cat a cat
        # Assesses if the truth image was predicted correctly
        c = decode_predictions(preds_true, top=3)
        list1, list2, list3 = zip(*c)
        c_class_1 = (list1[0][0])
        c_class_2 = (list2[0][0])
        c_class_3 = (list3[0][0])

        a = [*c_class_1]
        b = [*c_class_2]
        c = [*c_class_3]
        truth_trigger_normal = False

        # Determine if the truth prediction is true or false
        # if false, a new class is added
        truth_trigger_normal = True if a[2] in [b[2], c[2]] else None
        # Corroborates if the manipulated images are the same as the truth image
        valid = False
        invalid = False
        weak_pred = False
        if truth_trigger_normal is None:
            pass
        elif truth_trigger_normal == truth_trigger_dulled and truth_trigger_normal == truth_trigger_brightened:
            valid = True
        elif truth_trigger_normal not in [truth_trigger_dulled, truth_trigger_brightened]:
            invalid = True
        else:
            weak_pred = True

        if valid:
            num_valid += 1
        elif invalid:
            num_invalid += 1
        elif weak_pred:
            num_weak += 1
        else:
            num_pred_wrong += 1

    # Accuracy Calculation
    total_len = num_valid + num_invalid + num_weak + num_pred_wrong
    res_dict = {'percentage_valid': [num_valid / total_len], 'percentage_invalid': [num_invalid / total_len],
                'percentage_weak': [num_weak / total_len], 'percentage_wrong': [num_pred_wrong / total_len]}
    return pd.DataFrame(res_dict)


# coding=utf-8

import cv2 as cv2
import numpy as np
import os
import LBPDescriptor as LBP
import UniformLBPDescriptor as ULBP
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
import itertools
import cPickle
import warnings
import imutils
from itertools import product
from imutils.object_detection import non_max_suppression
import random
import string
import pandas as pd
import sys

PATH_POSITIVE_TRAIN = "../data/train/pedestrians/"
PATH_NEGATIVE_TRAIN = "../data/train/background/"
PATH_POSITIVE_TEST = "../data/test/pedestrians/"
PATH_NEGATIVE_TEST = "../data/test/background/"
PATH_MULTIPLE_PERSON = "../data/person_detection/"
EXAMPLE_POSITIVE = PATH_POSITIVE_TEST + "AnnotationsPos_0.000000_crop_000011b_0.png"
EXAMPLE_NEGATIVE = PATH_NEGATIVE_TEST + "AnnotationsNeg_0.000000_00000002a_0.png"


def __main__():
    ### Histogram of Gradients (HoG)
    #process(['hog'])
    ### Local binary pattern (LBP)
    process(['lbp'])
    ### Local Binary Pattern + Histogram of Gradients (LBP + HoG)
    #process(['lbp', 'hog'])
    ### Uniform Local Binary Pattern (ULBP)
    #process(['ulbp'])


def process(predictors):
    print "----> Loading data for " + ' '.join(predictors) + "predictors"
    data_train, data_test, classes_train, classes_test = load_data(predictors, orig_train_test=True)
    #data, classes = load_data(predictors, orig_train_test=False)
    print "----> SVM con parámetros estándar (" + ' '.join(predictors) + "):"
    print standard_svm(data_train, data_test, classes_train, classes_test, save=True, name="svm_std_" + '_'.join(predictors))
    print "----> SVM con 10-fold CV y parámetros estándar (" + ' '.join(predictors) + "):"
    #print cv_standard_svm(data, classes, save=True, name="svm_10cv_std_" + '_'.join(predictors))
    print "----> Búsqueda mejores parámetros SVM con 5-fold CV (" + ' '.join(predictors) + "):"
    #print find_best_params(data, classes, save=True, name="svm_5cv_grid_" + '_'.join(predictors))


def standard_svm(data_train, data_test, classes_train, classes_test, clf=None, save=False, name=None):
    clf = clf if clf is not None else train(data_train, classes_train)
    #if save: write_data(clf, '../clfs/' + name + ".pkl")
    prediction = test(data_test, clf)
    std_clf_metrics(classes_test, prediction, save=save, name=name)


def std_clf_metrics(classes_test, prediction, save=False, name=None):
    scores = {"Exactitud": [metrics.accuracy_score(classes_test, prediction)],
              "Precision": [metrics.precision_score(classes_test, prediction)],
              "Sensibilidad": [metrics.recall_score(classes_test, prediction)],
              "F1-Score": [metrics.f1_score(classes_test, prediction)]}
    df = pd.DataFrame.from_dict(scores)
    print df
    cm = (metrics.confusion_matrix(classes_test, prediction))
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_confusion_matrix(cm, normalize=False)
    '''
    if save:
        write_data(df, "../scores/" + name + ".pkl")
        write_data(cm, "../scores/" + name + "_cm.pkl")


def cv_standard_svm(data, classes, save=True, name=None):
    clf = svm.SVC(kernel='linear')
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(clf, data, classes, cv=10, n_jobs=-1, scoring=scoring, verbose=10, return_train_score=False)
    df = pd.DataFrame.from_dict(scores)
    if save: write_data(df, '../scores/' + name + ".pkl")
    return df


def find_best_params(data, classes, save=True, name=None):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 1, 10],
                         'C': [1, 10, 100]},
                        {'kernel': ['sigmoid'], 'gamma': [0.1, 1, 10],
                         'C': [1, 10, 100]},
                        {'kernel': ['linear'], 'C': [1, 10, 100]}]
    svc = svm.SVC()
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    clf = GridSearchCV(svc, tuned_parameters, cv=5, verbose=10, n_jobs=-1, return_train_score=False, scoring=scoring,
                       refit='accuracy')
    res = clf.fit(data, classes)
    df = pd.DataFrame.from_dict(res.cv_results_).iloc[:, 2:13]
    if save: write_data(df, '../scores/' + name + ".pkl")
    print df


def train(train_data, classes):
    idx = np.random.permutation(len(train_data))
    x, y = train_data[idx], classes[idx]
    clf = svm.SVC(kernel='rbf', gamma=0.1)

    clf.fit(x, y)
    return clf


def test(testData, clasificador):
    prediction = clasificador.predict(testData)
    return prediction


def load_data(descriptor_names, orig_train_test=False):
    switcher = {
        "hog": cv2.HOGDescriptor(),
        "lbp": LBP.LBPDescriptor(),
        "ulbp": ULBP.UniformLBPDescriptor()
    }
    descriptors = [switcher.get(descriptor_name) for descriptor_name in descriptor_names]
    data_train, classes_train = load_with_descriptor(PATH_POSITIVE_TRAIN, 1, descriptors)
    data_aux, classes_aux = load_with_descriptor(PATH_NEGATIVE_TRAIN, 0, descriptors)
    data_train = np.concatenate((data_train, data_aux), axis=0)
    classes_train = np.append(classes_train, classes_aux)
    data_test, classes_test = load_with_descriptor(PATH_POSITIVE_TEST, 1, descriptors)
    data_aux, classes_aux = load_with_descriptor(PATH_NEGATIVE_TEST, 0, descriptors)
    data_test = np.concatenate((data_test, data_aux), axis=0)
    classes_test = np.append(classes_test, classes_aux)

    if orig_train_test:
        return data_train, data_test, classes_train, classes_test
    else:
        return np.concatenate((data_train, data_test), axis=0), np.append(classes_train, classes_test)


def load_with_descriptor(path, label, descriptors):
    data = []
    classes = []
    lab = np.ones((1, 1), dtype=np.int32) if label == 1 else np.zeros((1, 1), dtype=np.int32)
    for file in os.listdir(path):
        print file
        img = cv2.imread(path + file, cv2.IMREAD_COLOR)
        img_d = [descriptor.compute(img).flatten() for descriptor in descriptors]
        data.append(np.concatenate(img_d))
        classes.append(lab)
    data = np.array(data)
    classes = np.array(classes, dtype=np.int32)
    return data, classes


def read_data(path):
    with open(path, 'rb') as fid:
        return cPickle.load(fid)


def write_data(clf, path):
    with open(path, 'wb') as fid:
        cPickle.dump(clf, fid)


def plot_confusion_matrix(cm, target_names=['Not_Person', 'Person'],
                          title='Confusion matrix', cmap=None, normalize=True):
    return "hola"
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

    def get_sliding_windows(image, stepSize, windowSize):
        coor = list(product(*[range(0, image.shape[0], stepSize), range(0, image.shape[1], stepSize)]))
        for y, x in coor: yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def get_image_resizes(image, scale=1.5, minSize=(30, 30)):
        yield image
        while True:
            image = imutils.resize(image, int(image.shape[1] / scale))
            if (image.shape[0] < minSize[1]) | (image.shape[1] < minSize[0]): break
            yield image

    def multi_target_person_detector(clf, descriptor=None):
        path_file = PATH_MULTIPLE_PERSON + 'padel.jpg'
        img = cv2.imread(path_file, cv2.IMREAD_COLOR)
        descriptor = cv2.HOGDescriptor()
        (winW, winH) = (64, 128)
        coors = []
        for resized in get_image_resizes(imutils.resize(img, int(img.shape[1] * 2)), scale=1.2):
            rt = img.shape[1] / float(resized.shape[1])
            (winW_r, winH_r) = (winW * rt, winH * rt)
            for (x, y, window) in get_sliding_windows(resized, stepSize=32, windowSize=(winW, winH)):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                part_img = resized[y:y + winH, x:x + winW]
                img_d = descriptor.compute(part_img)
                data = [img_d.flatten()]
                if clf.predict(data):
                    cv2.imwrite("../snaps/" + "".join(
                        random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) + ".jpg", part_img)
                    coor = [int(x * rt), int(y * rt), int(x * rt + winW_r), int(y * rt + winH_r)]
                    coors.append(coor)
        coors = np.array(coors)
        coors = non_max_suppression(coors, probs=None, overlapThresh=0.5)
        for x_s, y_s, x_e, y_e in coors:
            cv2.rectangle(img, (x_s, y_s), (x_e, y_e), (0, 255, 0), 2)
        cv2.imshow("Window", img)
        cv2.waitKey()
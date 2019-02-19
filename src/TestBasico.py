# coding=utf-8

import cv2 as cv2
import numpy as np
import os
import LBPDescriptor as LBP
import UniformLBPDescriptor as ULBP
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import itertools
import cPickle
import warnings
import imutils
from itertools import product
from imutils.object_detection import non_max_suppression
import argparse
import time
import random
import string


class TestBasico:
    PATH_POSITIVE_TRAIN = "../data/train/pedestrians/"
    PATH_NEGATIVE_TRAIN = "../data/train/background/"
    PATH_POSITIVE_TEST = "../data/test/pedestrians/"
    PATH_NEGATIVE_TEST = "../data/test/background/"
    PATH_MULTIPLE_PERSON = "../data/person_detection/"
    EXAMPLE_POSITIVE = PATH_POSITIVE_TEST + "AnnotationsPos_0.000000_crop_000011b_0.png"
    EXAMPLE_NEGATIVE = PATH_NEGATIVE_TEST + "AnnotationsNeg_0.000000_00000002a_0.png"

    def __main__(self):

        #self.generate_data(["lbp"])
        #self.generate_data(["ulbp"])
        #self.generate_data(["hog"])
        self.generate_data(["hog","lbp"])
        #data_train, data_test, classes_train, classes_test = self.load_data(["hog"],train_test=True)



    def get_sliding_windows(self, image, stepSize, windowSize):
        coor = list(product(*[range(0, image.shape[0], stepSize),range(0, image.shape[1], stepSize)]))
        for y,x in coor: yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def get_image_resizes(self, image, scale=1.5, minSize=(30, 30)):
        yield image
        while True:
            image = imutils.resize(image, int(image.shape[1] / scale))
            if (image.shape[0] < minSize[1]) | (image.shape[1] < minSize[0]): break
            yield image


    def multi_target_person_detector(self, clf, descriptor = None):

        path_file = self.PATH_MULTIPLE_PERSON +'padel.jpg'
        img = cv2.imread(path_file, cv2.IMREAD_COLOR)

        descriptor = cv2.HOGDescriptor()
        (winW, winH) = (64, 128)

        coors = []

        for resized in self.get_image_resizes(imutils.resize(img, int(img.shape[1] * 2)), scale=1.2):

            rt = img.shape[1] / float(resized.shape[1])
            (winW_r, winH_r) = (winW*rt, winH*rt)
            
            for (x, y, window) in self.get_sliding_windows(resized, stepSize=32, windowSize=(winW, winH)):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                part_img = resized[y:y+winH, x:x+winW]

                img_d = descriptor.compute(part_img)
                data = [img_d.flatten()]

                if clf.predict(data):
                    cv2.imwrite("../snaps/"+"".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) + ".jpg", part_img)
                    coor = [int(x * rt), int(y * rt), int(x * rt + winW_r), int(y * rt + winH_r)]
                    coors.append(coor)

        coors = np.array(coors)
        coors = non_max_suppression(coors, probs=None, overlapThresh=0.5)

        for x_s, y_s, x_e, y_e in coors:
            cv2.rectangle(img, (x_s, y_s), (x_e, y_e), (0, 255, 0), 2)

        cv2.imshow("Window", img)
        cv2.waitKey()


    def find_best_params(self, data, classes):
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(svc, parameters, cv=5, verbose=10)
        res = clf.fit(data, classes)
        print res

    def cv_standard_svm(self, data, classes, clf=None, save=False, name=None):
        clf = clf if clf != None else svm.SVC(kernel='linear', C=1)
        if save: self.save_clf(clf, '../clfs/' + name)
        print cross_val_score(clf, data, classes, cv=10, n_jobs=-1)

    def standard_svm(self, data_train, data_test, classes_train, classes_test, clf=None, save=False, name=None):
        clf = clf if clf != None else self.train(data_train, classes_train)
        if save: self.save_clf(clf, '../clfs/' + name)
        prediction = self.test(data_test, clf)
        self.metrics(classes_test, prediction)

    def metrics(self, classesTest, prediction):
        print "Exactitud: " + str(metrics.accuracy_score(classesTest, prediction))
        print "Precisión: " + str(metrics.precision_score(classesTest, prediction))
        print "Sensibilidad: " + str(metrics.recall_score(classesTest, prediction))
        print "F1-score: " + str(metrics.f1_score(classesTest, prediction))
        print "Matriz de confusión:"
        cm = (metrics.confusion_matrix(classesTest, prediction))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.plot_confusion_matrix(cm, normalize=False)

    def train(self, trainingData, classes):
        clf = svm.SVC(kernel='linear')
        clf.fit(trainingData, classes)
        return clf

    def test(self, testData, clasificador):
        prediccion = clasificador.predict(testData)
        return prediccion

    def generate_data(self, descriptor_names):
        switcher = {
            "hog": cv2.HOGDescriptor(),
            "lbp": LBP.LBPDescriptor(),
            "ulbp": ULBP.UniformLBPDescriptor()
        }
        descriptors = [switcher.get(descriptor_name) for descriptor_name in descriptor_names]

        dataTrain, classesTrain = self.load_with_descriptor(self.PATH_POSITIVE_TRAIN, 1, descriptors)
        data_aux, classes_aux = self.load_with_descriptor(self.PATH_NEGATIVE_TRAIN, 0, descriptors)
        dataTrain = np.concatenate((dataTrain, data_aux), axis=0)
        classesTrain = np.append(classesTrain, classes_aux)

        dataTest, classesTest = self.load_with_descriptor(self.PATH_POSITIVE_TEST, 1, descriptors)
        data_aux, classes_aux = self.load_with_descriptor(self.PATH_NEGATIVE_TEST, 0, descriptors)
        dataTest = np.concatenate((dataTest, data_aux), axis=0)
        classesTest = np.append(classesTest, classes_aux)

        data, classes = np.concatenate((dataTrain, dataTest), axis=0), np.append(classesTrain, classesTest)

        self.write_data(dataTrain, "../tidy_datasets/" + '_'.join(descriptor_names) + "_data_train.pkl")
        self.write_data(dataTest, "../tidy_datasets/" + '_'.join(descriptor_names) + "_data_test.pkl")
        self.write_data(classesTrain, "../tidy_datasets/" + '_'.join(descriptor_names) + "_classes_train.pkl")
        self.write_data(classesTest, "../tidy_datasets/" + '_'.join(descriptor_names) + "_classes_test.pkl")
        self.write_data(data, "../tidy_datasets/" + '_'.join(descriptor_names) + "_data.pkl")
        self.write_data(classes, "../tidy_datasets/" + '_'.join(descriptor_names) + "_classes.pkl")

    def load_data(self, descriptor_names, train_test=False):
        if train_test:
            data_train = self.read_data("../tidy_datasets/" + '_'.join(descriptor_names) + "_data_train.pkl")
            data_test = self.read_data("../tidy_datasets/" + '_'.join(descriptor_names) + "_data_test.pkl")
            clases_train = self.read_data("../tidy_datasets/" + '_'.join(descriptor_names) + "_classes_train.pkl")
            classes_test = self.read_data("../tidy_datasets/" + '_'.join(descriptor_names) + "_classes_test.pkl")
            return data_train, data_test, clases_train, classes_test
        else:
            data = self.read_data("../tidy_datasets/" + '_'.join(descriptor_names) + "_data.pkl")
            classes = self.read_data("../tidy_datasets/" + '_'.join(descriptor_names) + "_classes.pkl")
            return data, classes

    def load_with_descriptor(self, path, label, descriptors):
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

    def read_data(self, path):
        with open(path, 'rb') as fid:
            return cPickle.load(fid)

    def write_data(self, clf, path):
        with open(path, 'wb') as fid:
            cPickle.dump(clf, fid)

    def plot_confusion_matrix(self, cm, target_names=['Not_Person', 'Person'],
                              title='Confusion matrix', cmap=None, normalize=True):
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

        # print '----> Histogram of Gradients'
        # hog = cv2.HOGDescriptor()
        # data_train, data_test, classes_train, classes_test = self.load_data(hog, trainTest=True)

        # print "-> SVM con parámetros estándar (HoG)"

        # self.standard_svm(data_train, data_test, classes_train, classes_test, save=True, name='svm_std_HoG.pkl')

        # clf = self.load_clf('svm_std_HoG.pkl')
        # self.standard_svm(data_train, data_test, classes_train, classes_test, clf = clf)

        # self.standard_svm(clf=clf)

        '''
        
        clf = self.load_clf('svm_std_HoG.pkl')
        
        print "----> SVM 10-fold CV con parámetros estándar (HoG)"

        data, classes = self.load_data(hog)

        self.standard_svm(data, classes)


        print "----> SVM con mejores parámetros (HoG)"
        self.find_best_params(data, classes)


        lbp = ULBP.UniformLBPDescriptor()
        #dataTrain, dataTest, classesTrain, classesTest = self.load_data(lbp, trainTest=True)
        data, classes = self.load_data(lbp)

        print "----> SVM con parámetros estándar (HoG)"
        self.cv_standard_svm(data, classes)
        '''
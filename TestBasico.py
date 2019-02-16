# coding=utf-8

import cv2 as cv2
import numpy as np
import os
import sys as sys
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV


class TestBasico:

    def __main__(self):
        self.standardSVMMetrics()


    PATH_POSITIVE_TRAIN = "data/train/pedestrians/"
    PATH_NEGATIVE_TRAIN = "data/train/background/"
    PATH_POSITIVE_TEST = "data/test/pedestrians/"
    PATH_NEGATIVE_TEST = "data/test/background/"
    EXAMPLE_POSITIVE = PATH_POSITIVE_TEST + "AnnotationsPos_0.000000_crop_000011b_0.png"
    EXAMPLE_NEGATIVE = PATH_NEGATIVE_TEST + "AnnotationsNeg_0.000000_00000002a_0.png"

    def loadTrainingData(self):
        trainingData = []
        classes = []

        listFiles = os.listdir(self.PATH_POSITIVE_TRAIN)
        for file in os.listdir(self.PATH_POSITIVE_TRAIN):
            img = cv2.imread(self.PATH_POSITIVE_TRAIN + file, cv2.IMREAD_COLOR)
            hog = cv2.HOGDescriptor()
            descriptor = hog.compute(img)
            trainingData.append(descriptor.flatten())
            classes.append(np.ones((1, 1), dtype=np.int32))

        print("Leidas " + str(len(listFiles)) + " imágenes de entrenamiento -> positivas")

        listFiles = os.listdir(self.PATH_NEGATIVE_TRAIN)
        for file in os.listdir(self.PATH_NEGATIVE_TRAIN):
            img = cv2.imread(self.PATH_NEGATIVE_TRAIN + file, cv2.IMREAD_COLOR)
            hog = cv2.HOGDescriptor()
            descriptor = hog.compute(img)
            trainingData.append(descriptor.flatten())
            classes.append(np.zeros((1, 1), dtype=np.int32))

        print("Leidas " + str(len(listFiles)) + " imágenes de entrenamiento -> negativas")

        trainingData = np.array(trainingData)
        classes = np.array(classes, dtype=np.int32)

        return trainingData, classes

    def ejemploClasificadorImagenes(self):
        descriptor = cv2.HOGDescriptor()
        data, classes = self.loadData(descriptor)

        self.find_best_params(data,classes)
        clasificador = self.test_pred(data, classes)
        print clasificador

    def standardSVMMetrics(self):
        descriptor = cv2.HOGDescriptor()
        dataTrain, dataTest, classesTrain, classesTest = self.loadData(descriptor, trainTest=True)
        clasificador = self.train(dataTrain, classesTrain)
        prediccion = self.test(dataTest, clasificador)
        classesTest = np.reshape(classesTest, (classesTest.size,1))

        print "Predicción: " + str(metrics.accuracy_score(classesTest, prediccion))
        metrics.confusion_matrix()
        metrics.recall_score()
        metrics.f1_score()
        

    def loadData(self, descriptor, trainTest = False):
        dataTrain, classesTrain = self.loadWithDescriptor(self.PATH_POSITIVE_TRAIN, 1, descriptor)
        data_aux, classes_aux = self.loadWithDescriptor(self.PATH_NEGATIVE_TRAIN, 0, descriptor)
        dataTrain = np.concatenate((dataTrain, data_aux), axis=0)
        classesTrain = np.append(classesTrain, classes_aux)

        dataTest, classesTest = self.loadWithDescriptor(self.PATH_POSITIVE_TEST, 1, descriptor)
        data_aux, classes_aux = self.loadWithDescriptor(self.PATH_NEGATIVE_TEST, 0, descriptor)
        dataTest = np.concatenate((dataTest, data_aux), axis=0)
        classesTest = np.append(classesTest, classes_aux)

        if(trainTest):
            return dataTrain, dataTest, classesTrain, classesTest
        else:
            return np.concatenate((dataTrain,dataTest), axis=0), np.append(classesTrain,classesTest)




    def loadWithDescriptor(self, path, label, descriptor):
        data = []
        classes = []
        lab = np.ones((1, 1), dtype=np.int32) if label == 1 else np.zeros((1, 1), dtype=np.int32)
        for file in os.listdir(path):
            img = cv2.imread(path + file, cv2.IMREAD_COLOR)
            img_d = descriptor.compute(img)
            data.append(img_d.flatten())
            classes.append(lab)
        data = np.array(data)
        classes = np.array(classes, dtype=np.int32)
        return data, classes

    def test_pred(self, data, classes):
        clf = svm.SVC(kernel='linear')
        scores = cross_val_score(clf, data, classes, cv = 5)
        return scores

    def find_best_params(self, data, classes):
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(svc, parameters, cv=5, verbose=10)
        res = clf.fit(data, classes)
        print  res

    def test(self, testData, clasificador):
        prediccion = clasificador.predict(testData)
        return prediccion[1]

    def train(self, trainingData, classes):
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.train(trainingData, cv2.ml.ROW_SAMPLE, classes)
        return svm
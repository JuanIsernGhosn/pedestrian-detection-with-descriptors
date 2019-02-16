# coding=utf-8

import cv2 as cv2
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import itertools


class TestBasico:

    PATH_POSITIVE_TRAIN = "data/train/pedestrians/"
    PATH_NEGATIVE_TRAIN = "data/train/background/"
    PATH_POSITIVE_TEST = "data/test/pedestrians/"
    PATH_NEGATIVE_TEST = "data/test/background/"
    EXAMPLE_POSITIVE = PATH_POSITIVE_TEST + "AnnotationsPos_0.000000_crop_000011b_0.png"
    EXAMPLE_NEGATIVE = PATH_NEGATIVE_TEST + "AnnotationsNeg_0.000000_00000002a_0.png"

    def __main__(self):
        hog = cv2.HOGDescriptor()
        dataTrain, dataTest, classesTrain, classesTest = self.load_data(hog, trainTest=True)
        data, classes = self.load_data(hog)

        print "----> SVM con parámetros estándar (HoG)"
        self.standard_svm(data, classes)

        print "----> SVM 10-fold CV con parámetros estándar (HoG)"
        self.cv_standard_svm(dataTrain, dataTest, classesTrain, classesTest)

        print "----> SVM con mejores parámetros (HoG)"
        self.find_best_params(data, classes)

    def cv_standard_svm(self, data, classes):
        clf = svm.SVC(kernel='linear', C=1)
        print "Precisión de los folds: " + str(cross_val_score(clf, data, classes, cv=10))

    def standard_svm(self, dataTrain, dataTest, classesTrain, classesTest):
        clf = self.train(dataTrain, classesTrain)
        prediction = self.test(dataTest, clf)
        classesTest = np.reshape(classesTest, (classesTest.size, 1))
        self.metrics(classesTest, prediction)

    def metrics(self, classesTest, prediction):
        print "Precisión: " + str(metrics.accuracy_score(classesTest, prediction))
        print "Sensibilidad: " + str(metrics.recall_score(classesTest, prediction))
        print "F1-score: " + str(metrics.f1_score(classesTest, prediction))
        print "Matriz de confusión:"
        cm = (metrics.confusion_matrix(classesTest, prediction))
        self.plot_confusion_matrix(cm, normalize=False)

    def train(self, trainingData, classes):
        clf = svm.SVC(kernel='linear')
        clf.fit(trainingData, classes)
        return clf

    def test(self, testData, clasificador):
        prediccion = clasificador.predict(testData)
        return prediccion

    def find_best_params(self, data, classes):
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(svc, parameters, cv=5, verbose=10)
        res = clf.fit(data, classes)
        print res

    def load_data(self, descriptor, trainTest=False):
        dataTrain, classesTrain = self.load_with_descriptor(self.PATH_POSITIVE_TRAIN, 1, descriptor)
        data_aux, classes_aux = self.load_with_descriptor(self.PATH_NEGATIVE_TRAIN, 0, descriptor)
        dataTrain = np.concatenate((dataTrain, data_aux), axis=0)
        classesTrain = np.append(classesTrain, classes_aux)

        dataTest, classesTest = self.load_with_descriptor(self.PATH_POSITIVE_TEST, 1, descriptor)
        data_aux, classes_aux = self.load_with_descriptor(self.PATH_NEGATIVE_TEST, 0, descriptor)
        dataTest = np.concatenate((dataTest, data_aux), axis=0)
        classesTest = np.append(classesTest, classes_aux)

        if (trainTest):
            return dataTrain, dataTest, classesTrain, classesTest
        else:
            return np.concatenate((dataTrain, dataTest), axis=0), np.append(classesTrain, classesTest)

    def load_with_descriptor(self, path, label, descriptor):
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
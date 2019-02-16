# coding=utf-8

import cv2 as cv2
import numpy as np
import os
import sys as sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class TestBasico:

    def __main__(self):
        print "Versión OpenCV "+ cv2.__version__

        # Leemos imagen
        strImgPrueba = "Fry.jpg"
        image = cv2.imread(strImgPrueba, cv2.IMREAD_COLOR)
        height, width, channels = image.shape
        print "Tamaño imagen " + str(height) + "x" + str(width) + " y canales " + str(channels)

        #  Obtenemos una versión en nivel de gris y otra suavizada
        gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.blur(image, (3,3))

        # 1. - ----- Canny
        canny = cv2.Canny(blur, 10, 100)

        # 2. ------ Harris
        # La API dice que src tiene que ser flotante, pero no da excepción
        harris = cv2.cornerHarris(gris, 2, 3, 0.04)

        # Imgproc.dilate(harris, harris, new Mat());
        # Resalta(no necesario)
        # Para visualizar, normalizo entre 0 y 255
        harris = cv2.normalize(harris, 0, 255, cv2.NORM_MINMAX)
        harris = cv2.convertScaleAbs(harris)

        # 3. ------ HOG
        hog = cv2.HOGDescriptor()
        descriptors = hog.compute(image)

        print ("HOG (" + str(descriptors.size) + "): "
        + "block size: " + str(hog.blockSize)
        + ", window size: " + str(hog.winSize)
        + ", stride size: " + str(hog.blockStride)
        + ", cell size: " + str(hog.cellSize)
        + ", number of bins: " + str(hog.nbins)
        + ", descriptor size: " + str(hog.getDescriptorSize()))


        # 4. ------ SIFT
        fd = cv2.FastFeatureDetector.create()
        sift = fd.detect(gris)
        #siftImage = cv2.drawKeypoints(gris, sift)

        # Mostramos resultados
        cv2.namedWindow(strImgPrueba, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(strImgPrueba, image)
        cv2.namedWindow("Canny", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Canny", canny)
        cv2.namedWindow("Harris", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Harris", harris)
        cv2.namedWindow("SIFT", cv2.WINDOW_AUTOSIZE)
        #cv2.imshow("SIFT", siftImage)

        # 5. - ----- Clasificación
        self.ejemploClasificadorImagenes()


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
        trainingData, classes = self.loadTrainingData()
        print "es" + str(trainingData.shape)
        clasificador = self.train(trainingData, classes)
        print "Clasificador entrenado"

        testData, testClasses = self.loadTestData()

        prediccion = self.test(testData, clasificador, testClasses)
        print "Predicción: " + str(prediccion)

    def loadTestData(self):
        testData = []
        classes = []

        listFiles = os.listdir(self.PATH_POSITIVE_TEST)
        for file in os.listdir(self.PATH_POSITIVE_TEST):
            img = cv2.imread(self.PATH_POSITIVE_TEST + file, cv2.IMREAD_COLOR)
            hog = cv2.HOGDescriptor()
            descriptor = hog.compute(img)
            testData.append(descriptor.flatten())
            classes.append(np.ones((1, 1), dtype=np.int32))

        print("Leidas " + str(len(listFiles)) + " imágenes de test -> positivas")

        listFiles = os.listdir(self.PATH_NEGATIVE_TEST)
        for file in os.listdir(self.PATH_NEGATIVE_TEST):
            img = cv2.imread(self.PATH_NEGATIVE_TEST + file, cv2.IMREAD_COLOR)
            hog = cv2.HOGDescriptor()
            descriptor = hog.compute(img)
            testData.append(descriptor.flatten())
            classes.append(np.zeros((1, 1), dtype=np.int32))

        print("Leidas " + str(len(listFiles)) + " imágenes de test -> negativas")

        testData = np.array(testData)
        classes = np.array(classes, dtype=np.int32)

        return testData, classes

    def train(self, trainingData, classes):
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.train(trainingData, cv2.ml.ROW_SAMPLE, classes)
        return svm

    def test(self, testData, clasificador,testClasses):
        prediccion = clasificador.predict(testData)
        #accuracy = accuracy_score(testClasses, np.array(prediccion[1]))
        return prediccion
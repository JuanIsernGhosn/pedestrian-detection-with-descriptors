# coding=utf-8

import cv2 as cv2
import numpy as np
import os
import LBPDescriptor as LBP
import UniformLBPDescriptor as ULBP
from sklearn import metrics, svm
from sklearn.model_selection import cross_validate, GridSearchCV
import cPickle
import imutils
from itertools import product
import pandas as pd
import warnings
import matplotlib.pyplot as plt

PATH_POSITIVE_TRAIN = "../data/train/pedestrians/"
PATH_NEGATIVE_TRAIN = "../data/train/background/"
PATH_POSITIVE_TEST = "../data/test/pedestrians/"
PATH_NEGATIVE_TEST = "../data/test/background/"
PATH_MULTIPLE_PERSON = "../data/person_detection/"
EXAMPLE_POSITIVE = PATH_POSITIVE_TEST + "AnnotationsPos_0.000000_crop_000011b_0.png"
EXAMPLE_NEGATIVE = PATH_NEGATIVE_TEST + "AnnotationsNeg_0.000000_00000002a_0.png"


def __main__():
    '''
    ### Histogram of Gradients (HoG)
    process(['hog'])
    ### Local binary pattern (LBP)
    process(['lbp'])
    ### Uniform Local Binary Pattern (ULBP)
    process(['ulbp'])
    ### Local Binary Pattern + Histogram of Gradients (LBP + HoG)
    process(['lbp', 'hog'])
    '''

    ## Multiple person detection
    #  clf = get_custom_SVM(['ulbp'], kernel='rbf', gamma=0.01, C=10)

    clf = read_data("../clfs/ulbp.pkl")

    multi_target_person_detector(clf, ULBP.UniformLBPDescriptor())


def get_custom_SVM(descriptors, gamma=None, C=1, kernel='linear'):
    """Get SVM model for given parameters and descriptor(s)
    
    Args:
        (String[]) descriptors: Descriptors 
        (float) gamma: SVM gamma parameter (Default = None)
        (float) C: SVM C parameter (Default = 1)
        (String) kernel: SVM kernel (Default = 'linear')
    """
    data, classes = load_data(descriptors, orig_train_test=False)
    idx = np.random.permutation(len(data))
    x, y = data[idx], classes[idx]
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, probability=True)
    clf.fit(x, y)
    print "--> Clasificador entrenado"
    return clf


def process(descriptors):
    """Execute process for:
    
    1.  Calculate standard SVM and performance metrics
    2.  Calculate standard SVM and 10-fold CV performance metrics
    3.  Parameter search grid for SVM and performance metrics

    Args:
        (String[]) descriptors: Image descriptors for making process. 
    """

    print "----> Loading data for " + ' '.join(descriptors) + "predictors"
    data_train, data_test, classes_train, classes_test = load_data(descriptors, orig_train_test=True)
    data, classes = load_data(descriptors, orig_train_test=False)

    print "----> SVM con parámetros estándar (" + ' '.join(descriptors) + "):"
    print standard_svm(data_train, data_test, classes_train, classes_test, save=True,
                       name="svm_std_" + '_'.join(descriptors))

    print "----> SVM con 10-fold CV y parámetros estándar (" + ' '.join(descriptors) + "):"
    print cv_standard_svm(data, classes, save=True, name="svm_10cv_std_" + '_'.join(descriptors))

    print "----> Búsqueda mejores parámetros SVM con 5-fold CV (" + ' '.join(descriptors) + "):"
    print find_best_params(data, classes, save=True, name="svm_5cv_grid_" + '_'.join(descriptors))


def standard_svm(data_train, data_test, classes_train, classes_test, save=False, name=None):
    """Compute standard linear SVM classifier and metrics for a given train-test data set.

    Compute Accuracy, precision, recall, F1 and print confusion 
    matrix for Linear standard SVM and given train-test data set.

    Args:
        (float[][]) data_train: Image descriptors of train data.
        (float[][]) data_test: Image descriptors of test data.
        (int[]) classes_train: Real label of the train data.
        (int[]) classes_test: Real label of the test data.
        (bool) save: Save CV score.
        (String) name: name of score file. 
    """
    clf = train(data_train, classes_train)
    prediction = test(data_test, clf)
    std_clf_metrics(classes_test, prediction, save=save, name=name)


def std_clf_metrics(classes_test, prediction, save=False, name=None):
    """Compute metrics for a given prediction.

    Compute Accuracy, precision, recall, F1 and print confusion 
    matrix for given prediction.

    Args:
        (int[]) classes_test: Real label of the data.
        (int[]) prediction: Predicted label of the data.
        (bool) save: Save CV score.
        (String) name: name of score file. 
    """
    scores = {"Exactitud": [metrics.accuracy_score(classes_test, prediction)],
              "Precision": [metrics.precision_score(classes_test, prediction)],
              "Sensibilidad": [metrics.recall_score(classes_test, prediction)],
              "F1-Score": [metrics.f1_score(classes_test, prediction)]}
    df = pd.DataFrame.from_dict(scores)
    print(df)
    cm = (metrics.confusion_matrix(classes_test, prediction))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_confusion_matrix(cm, normalize=False)
    if save:
        write_data(df, "../scores/" + name + ".pkl")
        write_data(cm, "../scores/" + name + "_cm.pkl")


def cv_standard_svm(data, classes, save=True, name=None):
    """Compute CV with default parameters.

    Compute cross-validation for standard linear SVM classifier.

    Args:
        (float[][]) data: Image descriptors.
        (int[]) classes: Label of the data.
        (bool) save: Save CV score. 
        (String) name: name of score file. 
        (int) cv: Number of folds.
    Returns:
        (DataFrame): Scores for each CV split.
    """
    clf = svm.SVC(kernel='linear')
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(clf, data, classes, cv=10, n_jobs=-1, scoring=scoring, verbose=10, return_train_score=False)
    df = pd.DataFrame.from_dict(scores)
    if save: write_data(df, '../scores/' + name + ".pkl")
    return df


def find_best_params(data, classes, save=True, name=None):
    """SVM parameter tuning.

    Search best kernel and parameters for SVM classifier.

    Args:
        (float[][]) data: Image descriptors.
        (int[]) classes: Label of the data.
        (bool) save: Save CV score. 
        (String) name: name of score file. 
    Returns:
        (DataFrame): Scores for kernel & parameter combination.
    """
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1],
                         'C': [1, 10, 100]},
                        {'kernel': ['sigmoid'], 'gamma': [0.001, 0.01, 0.1],
                         'C': [1, 10, 100]},
                        {'kernel': ['linear'], 'C': [1, 10, 100]}]
    svc = svm.SVC()
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    clf = GridSearchCV(svc, tuned_parameters, cv=5, verbose=10, n_jobs=-1,
                       return_train_score=False, scoring=scoring,
                       refit='accuracy')
    res = clf.fit(data, classes)
    df = pd.DataFrame.from_dict(res.cv_results_).iloc[:, 2:13]
    print df
    if save: write_data(df, '../scores/' + name + ".pkl")


def train(data_train, classes):
    """Compute standard linear SVM classifier.

    Args:
        (float[][]) data_train: Image descriptors of train data.
        (int[]) classes: Label of the train data.
    Returns:
        (SVM): Trained SVM classifier.
    """
    idx = np.random.permutation(len(data_train))
    x, y = data_train[idx], classes[idx]
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    return clf


def test(data_test, classifier):
    """Compute predictions for a given classifier.

    Args:
        (float[][]) data_test: Image descriptors of test data.
        (SVM) classifier: SVM classifier to use. 
    Returns:
        (int[]): Predictions for given data and classifier.
    """
    prediction = classifier.predict(data_test)
    return prediction


def load_data(descriptor_names, orig_train_test=False):
    """Load image descriptors data set.

    Args:
        (String[]) descriptor_names: Name of the descriptors
        to use for loading each image.
        (bool) orig_train_test: Keep original train/test split.
    Returns:
        (float[][]) data_train: Image descriptors of train data.
        (float[][]) data_test: Image descriptors of test data.
        (int[]) classes_train: Labels for train images.
        (int[]) classes_test: Labels for test images.
    """
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
    """Load image descriptors data set.

    Args:
        (String) path: Path of the images folder.
        (int) label: Label of the images of the given path.
        (Descriptor[]) descriptors: Descriptors to use.
    Returns:
        (float[][]) data: Image descriptors of the data.
        (int[]) classes: Labels for images.
    """
    data = []
    classes = []
    lab = np.ones((1, 1), dtype=np.int32) if label == 1 else np.zeros((1, 1), dtype=np.int32)
    for file in os.listdir(path):
        if file.startswith('.'): continue
        img = cv2.imread(path + file, cv2.IMREAD_COLOR)
        img_d = [descriptor.compute(img).flatten() for descriptor in descriptors]
        data.append(np.concatenate(img_d))
        classes.append(lab)
        print file
    data = np.array(data)
    classes = np.array(classes, dtype=np.int32)
    return data, classes


def read_data(path):
    """Read .pkl file saved locally.
    
    Args:
        (String) path: Path of the file to load.
    Returns:
        (Object): Read file object.
         
    """
    with open(path, 'rb') as fid:
        return cPickle.load(fid)


def write_data(clf, path):
    """Save .pkl file locally.
    
    Args:
        (Object): Object to be saved.
    """
    with open(path, 'wb') as fid:
        cPickle.dump(clf, fid)


def plot_confusion_matrix(cm, target_names=['Not_Person', 'Person'],
                          title='Confusion matrix', normalize=True):
    """Plot confusion matrix with better aesthetic.

    Args:
        (int[][]) cm: Confusion matrix to plot.
        (String[]) target_names: Name of the matrix labels
        (String) title: Title to print above the matrix.
        (bool) normalize: Set if is matrix normalization required.
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

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
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
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


def get_window_coor(image, step, w_size):
    """Get sliding window coordinates for an given image,
    window and step size.

    Args:
        (numeric[][]) image: Image where to compute sliding windows 
        coordinates.
        (int) step: Pixel difference between windows.
        (int[]) w_size: Size of the required windows.
    Returns:
        (int[]): Coordinates for each window.
    """
    coor = list(product(*[range(0, image.shape[0], step), range(0, image.shape[1], step)]))
    for y, x in coor: yield (x, y, image[y:y + w_size[1], x:x + w_size[0]])


def get_resizes(image, scale=1.5, min_size=(64, 128)):
    """Get different image resizes of an original  image.
    
    Args:
        (numeric[][]) image: Image to resize.
        (float) scale: factor of resize.
        (int[]) min_size: Min size of the resized image.
    Returns:
        (numeric[][]): Resized images.
    """
    while True:
        yield image
        if (image.shape[0] < min_size[1]) or (image.shape[1] < min_size[0]): break
        else: image = imutils.resize(image, int(image.shape[1] / scale))


def multi_target_person_detector(clf, descriptor):
    """Detect multiple person in images contained in data/person_detection.

    Args:
        (SVM) clf: Classifier to use for detection.
        (Descriptor) descriptor: descriptor to use for image feature extraction.
    """
    for file in os.listdir(PATH_MULTIPLE_PERSON):
        if file.startswith('.'): continue
        print file
        img = cv2.imread(PATH_MULTIPLE_PERSON + file, cv2.IMREAD_COLOR)
        person_detector(clf, img, descriptor, file)
    cv2.waitKey(0)


def person_detector(clf, img, descriptor, file):
    """Detect multiple person in an image.

    Args:
        (SVM) clf: Classifier to use for detection.
        (numeric[][]) img: Image to be used for detection.
        (Descriptor) descriptor: descriptor to use for image feature extraction.
        (String) file: Name of the file in which to find multiple person.
    """
    (win_w, win_h) = (64, 128)
    coors = []
    probs = []

    for resized in get_resizes(imutils.resize(img, int(img.shape[1] * 2)), scale=1.5):
        rt = img.shape[1] / float(resized.shape[1])
        (win_w_r, win_h_r) = (win_w * rt, win_h * rt)

        for (x, y, window) in get_window_coor(resized, step=32, w_size=(win_w, win_h)):
            if window.shape[0] != win_h or window.shape[1] != win_w: continue
            img_d = descriptor.compute(window)
            data = [img_d.flatten()]

            prob = clf.predict_proba(data)[0, 1]

            if prob > 0.7:
                coor = [int(x * rt), int(y * rt), int(x * rt + win_w_r), int(y * rt + win_h_r)]
                coors.append(coor)
                probs.append(prob)

    boxes = non_max_suppression_fast(np.array(coors), 0.3)

    for x_s, y_s, x_e, y_e in boxes:
        cv2.rectangle(img, (x_s, y_s), (x_e, y_e), (0, 255, 0), 2)

    cv2.namedWindow("Person detection_" + file, cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Person detection_" + file, img)
    cv2.waitKey(1)


def non_max_suppression_fast(boxes, overlap_thresh):
    """Extracted from Malisiewicz et al. 
    (https://github.com/quantombone/exemplarsvm)
    
    Remove overlapped bounding boxes in an image. 

    Args:
        (int[][]) boxes: Bounding boxes coordinates.
        (float) overlap_thresh: Thresh to remove two overlapped images.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

import argparse
import csv
import distutils.util
import os
import Dummy
from PIL import Image
import numpy as np
from scipy.spatial.distance import euclidean
import cv2
import math
from skimage.metrics import structural_similarity


#  In this file please complete the following tasks:
#
#  Task 1 [10] My first not-so-pretty image classifier
#
# By using the kNN approach and three distance or similarity measures, build image classifiers.
# •	You must implement the kNN approach yourself
# •	You must invoke the distance or similarity measures from libraries (it is fine to invoke different measures
# from one library)
# •	Histogram-based measures are not allowed
#
# The classifier is expected to use only one measure at a time and take information as to which one to invoke at a
# given time as input. The template contains a range of functions you need to implement for this task.
#
#
#  Task 5 [4] Similarities
#
# Independent inquiry time! In Task 1, you were instructed to use libraries for image similarity measures. Pick two
# of the three measures you have used and implement them yourself. You are allowed to use libraries to e.g.,
# calculate the root, power, average or standard deviation of some set (but, for example, numpy.linalg.norm is not
# permitted). The template contains a range of functions you need to implement for this task.
#
# Disclaimer: if you decide to implement MSE, do not implement RMSE.

# Please replace with your student id, including the "c" at the beginning!!!
student_id = 'c2100374'

# This is the classification scheme you should use for kNN
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


# In this function, please implement validation of the data that is supplied to or produced by the kNN classifier.
#
# INPUT:  data              : a list of lists that was read from the training data or data to classify csv
#                             (see parse_arguments function) or produced by the kNN function
#         predicted         : a boolean value stating whether the "PredictedClass" column should be present
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass"] if predicted variable
#                             is False and ["Path", "ActualClass", "PredictedClass"] if it is True
#                             (there can be more column names, but at least these three at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" column (if there are any) are classes from scheme
#                             AND (if predicted is True) the values in the "PredictedClass" column (if there are any)
#                             are classes from scheme
#                             AND there are as many Path entries as ActualClass (and PredictedClass, if predicted
#                             is True) entries
#
#                             False otherwise

def validateDataFormat(data, predicted):
    formatCorrect = False
    
    # check if headers are correct
    if predicted:
        if data[0][:3] == ["Path", "ActualClass", "PredictedClass"]:
            formatCorrect = True
    else:
        if data[0][:2] == ["Path", "ActualClass"]:
            formatCorrect = True
    
    # check if file paths are correct
    for row in data[1:]:
        if not os.path.isfile(row[0]):
            formatCorrect = False
            break
    
    # check if classes are correct
    for row in data[1:]:
        if row[1] not in classification_scheme:
            formatCorrect = False
            break
        if predicted and len(row) == 3 and row[2] not in classification_scheme:
            formatCorrect = False
            break
    
    # check if number of entries are correct
    if predicted:
        actualClassRows = [row for row in data[1:] if len(row) == 3]
        predictedClassRows = [row for row in data[1:] if len(row) == 4]
        if len(actualClassRows) == len(predictedClassRows) and len(data[1:]) == len(actualClassRows):
            formatCorrect = True
    else:
        actualClassRows = [row for row in data[1:] if len(row) == 2]
        if len(data[1:]) == len(actualClassRows):
            formatCorrect = True
    
    # print the "PredictedClass" column along with the original columns
    if predicted and formatCorrect:
        data[0].append("PredictedClass")
        for i in range(1, len(data)):
            predicted_class = data[i][2] if len(data[i]) == 3 else ""
            data[i].append(predicted_class)
    
    return formatCorrect


# This function does reading and resizing of an image located in a give path on your drive.
# DO NOT REMOVE ANY CHANNELS. DO NOT MODIFY PATHS.
#
# INPUT:  imagePath         : path to image. DO NOT MODIFY - take from the file as-is.
#         width, height     : dimensions to which you are asked to resize your image
#
# OUTPUT: image             : read and resized image, or empty list if the image is not found at a given path

def readAndResize(image_path, width=60, height=30):
    try:
        # Open image and resize
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # convert to RGB format if necessary
            img = img.resize((width, height))
            # Convert image to numpy array
            image = np.asarray(img)
        return image
    except FileNotFoundError:
        # If file not found, return None
        return None

# These functions compute the distance or similarity value between two images according to a particular
# similarity or distance measure. Return nan if images are empty. These three measures must be
# computed by libraries according to portfolio requirements.
#
# INPUT:  image1, image2    : two images to compare
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def computeMeasure1(image1, image2):
    if image1 is None or image2 is None:
        return float("inf")
    else:
        return euclidean(image1.flatten(), image2.flatten())


def computeMeasure2(image1, image2):
    # Compute Manhattan distance between image1 and image2
    if image1 is None or image2 is None:
        return math.nan
    
    return np.sum(np.abs(image1 - image2))

def computeMeasure3(image1, image2):
    # Return nan if either image is None
    if image1 is None or image2 is None:
        return float('nan')
    
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    (score, _) = structural_similarity(image1_gray, image2_gray, full=True)
    return score

# These functions compute the distance or similarity value between two images according to a particular similarity or
# distance measure. Return nan if images are empty. As name suggests, selfComputeMeasure 1 has to be your own
# implementation of the measure you have used in computeMeasure1 (same for 2). These two measures cannot be computed by
# libraries according to portfolio requirements.
#
# INPUT:  image1, image2    : two images to compare
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def selfComputeMeasure1(image1, image2):
    # Structural Similarity Index (SSIM)
    if not image1 or not image2:
        return float('nan')
    
    k1 = 0.01
    k2 = 0.03
    L = 255.0
    
    def compute_mu(image):
        return sum(sum(pixel) for pixel in image) / (len(image) * len(image[0]))
    
    def compute_sigma(image, mu):
        return math.sqrt(sum((pixel - mu) ** 2 for row in image for pixel in row) / (len(image) * len(image[0])))
    
    def compute_covariance(image1, image2, mu1, mu2):
        cov_sum = 0.0
        for i in range(len(image1)):
            for j in range(len(image1[i])):
                cov_sum += (image1[i][j] - mu1) * (image2[i][j] - mu2)
        return cov_sum / (len(image1) * len(image1[0]))
    
    mu1 = compute_mu(image1)
    mu2 = compute_mu(image2)
    sigma1 = compute_sigma(image1, mu1)
    sigma2 = compute_sigma(image2, mu2)
    sigma12 = compute_covariance(image1, image2, mu1, mu2)
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2.0
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
    ssim = (ssim + c3) / (1 + c3)
    return ssim

def selfComputeMeasure2(image1, image2):
    # Mean Squared Error (MSE)
    if not image1 or not image2:
        return float('nan')

    mse = 0.0
    for i in range(len(image1)):
        for j in range(len(image1[i])):
            mse += (image1[i][j] - image2[i][j]) ** 2

    mse /= (len(image1) * len(image1[0]))
    return mse


# This function is supposed to return a dictionary of classes and their occurrences as taken from k nearest neighbours.
#
# INPUT:  measure_classes   : a list of lists that contain two elements each - a distance/similarity value
#                             and class from scheme
#         k                 : the value of k neighbours
#         similarity_flag   : a boolean value stating that the measure used to produce the values above is a distance
#                             (False) or a similarity (True)
# OUTPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
def getClassesOfKNearestNeighbours(measures_classes, k, similarity_flag):
    nearest_neighbours_classes = {}
    
    # Sort the measures_classes list by distance/similarity value
    sorted_measures_classes = sorted(measures_classes, key=lambda x: x[0], reverse=similarity_flag)

    # Iterate over the k nearest neighbours and count their occurrences in the dictionary
    for i in range(k):
        _, class_name = sorted_measures_classes[i]
        if class_name not in nearest_neighbours_classes:
            nearest_neighbours_classes[class_name] = 0
        nearest_neighbours_classes[class_name] += 1

    return nearest_neighbours_classes


# Given a dictionary of classes and their occurrences, returns the most common class. In case there are multiple
# candidates, it follows the order of classes in the scheme. The function returns empty string if the input dictionary
# is empty, does not contain any classes from the scheme, or if all classes in the scheme have occurrence of 0.
#
# INPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
# OUTPUT: winner            : the most common class from the classification scheme. In case there are
#                             multiple candidates, it follows the order of classes in the scheme. Returns empty string
#                             if the input dictionary is empty, does not contain any classes from the scheme,
#                             or if all classes in the scheme have occurrence of 0
#

def getMostCommonClass(nearest_neighbours_classes):
    scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food'] # classes in the classification scheme
    max_occurrence = 0 # keeps track of the highest occurrence seen so far
    candidates = [] # list of candidate classes with the highest occurrence
    
    # iterate over classes in the classification scheme
    for c in classification_scheme:
        if c in nearest_neighbours_classes:
            occurrence = nearest_neighbours_classes[c]
            if occurrence > max_occurrence: # found a new maximum
                max_occurrence = occurrence
                candidates = [c]
            elif occurrence == max_occurrence: # found another candidate
                candidates.append(c)
                
    # select the winner
    if max_occurrence == 0 or not candidates: # no winner
        winner = ''
    elif len(candidates) == 1: # unique winner
        winner = candidates[0]
    else: # multiple candidates, follow the order of classes in the scheme
        for c in classification_scheme:
            if c in candidates:
                winner = c
                break               
    return winner


# In this function I expect you to implement the kNN classifier. You are free to define any number of helper functions
# you need for this! You need to use all of the other functions in the part of the template above.
#
# INPUT:  training_data       : a list of lists that was read from the training data csv (see parse_arguments function)
#         k                   : the value of k neighbours
#         measure_func        : the function to be invoked to calculate similarity/distance (any of the above)
#         similarity_flag     : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#         data_to_classify    : a list of lists that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
#     most_common_class_func  : the function to be invoked to find the most common class among the neighbours
#                             (by default, it is the one from above)
# get_neighbour_classes_func  : the function to be invoked to find the classes of nearest neighbours
#                             (by default, it is the one from above)
#         read_func           : the function to be invoked to find to read and resize images
#                             (by default, it is the one from above)
#  OUTPUT: classified_data    : a list of lists which expands the data_to_classify with the results on how your
#                             classifier has classified a given image. In case no classification can be performed due
#                             to absence of training_data or data_to_classify, it only contains the header list.


def kNN(training_data, k, measure_func, similarity_flag, data_to_classify,
        most_common_class_func=getMostCommonClass, get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=readAndResize):
    # This sets the header list
    classified_data = [('Path', 'ActualClass', 'PredictedClass')]
    # Define helper function to get the class of an image
    def getClass(image_path):
        image = read_func(image_path)
        if not image:
            return ''
        distances = []
        for row in training_data:
            training_image_path, training_image_class = row
            training_image = read_func(training_image_path)
            if not training_image:
                continue
            distance = measure_func(image, training_image)
            distances.append((distance, training_image_class))
        if not distances:
            # Return the most common class in the training data
            return most_common_class_func()
        else:    
            distances.sort(key=lambda x: x[0], reverse=not similarity_flag)
            k_nearest_neighbours = distances[:k]
            nearest_neighbours_classes = get_neighbour_classes_func(k_nearest_neighbours, k)
            return most_common_class_func(nearest_neighbours_classes)
    
    # Classify the images in data_to_classify
    for row in data_to_classify:
        image_path, actual_class = row
        predicted_class = getClass(image_path)
        classified_data.append((image_path, actual_class, predicted_class))
    
    return classified_data



##########################################################################################
# Do not modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.

def main():
    opts = parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = readCSVFile(opts['training_data'])
    data_to_classify = readCSVFile(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    print(opts['simflag'])
    result = kNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], data_to_classify,
                 eval(opts['mcc']), eval(opts['gnc']), eval(opts['rrf']))
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        writeCSVFile(out, result)


# Straightforward function to read the data contained in the file "filename"
def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def writeCSVFile(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -measure        : function to compute a given similarity/distance measure
#       -simflag        : flag telling us whether the above measure is a distance (False) or similarity (True)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)
#       mcc, gnc, rrf, vf,cf,sf,al
#                       : staff variables, do not use
#
def parseArguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-m', '--measure')
    parser.add_argument('-s', '--simflag', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-classified', type=str)
    parser.add_argument('-mcc', default="getMostCommonClass")
    parser.add_argument('-gnc', default="getClassesOfKNearestNeighbours")
    parser.add_argument('-rrf', default="readAndResize")
    parser.add_argument('-cf', default="confusionMatrix")
    parser.add_argument('-sf', default="splitDataForCrossValidation")
    parser.add_argument('-al', default="Task_1_5.kNN")
    params = parser.parse_args()

    opt = {'k': params.k,
           'f': params.f,
           'measure': params.measure,
           'simflag': params.simflag,
           'training_data': params.train,
           'data_to_classify': params.test,
           'classified_data': params.classified,
           'mode': params.unseen,
           'mcc': params.mcc,
           'gnc': params.gnc,
           'rrf': params.rrf,
           'cf': params.cf,
           'sf': params.sf,
           'al': params.al
           }
    return opt


if __name__ == '__main__':
    main()

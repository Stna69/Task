# In this file please complete the following task:
#
# Task 2 [4] Basic evaluation
#
# Evaluate your classifiers. On your own, implement a method that will create a confusion matrix based on the
# provided classified data. Then implement methods that will output precision, recall, F-measure, and accuracy of
# your classifier based on your confusion matrix. Use macro-averaging approach and be mindful of edge cases. The
# template contains a range of functions you need to implement for this task.
#
# You are expected to rely on solutions from Task_1_5 here!

import Task_1_5
import Dummy
import numpy as np

# This function computes the confusion matrix based on the provided data.
#
# INPUT: classified_data   : a list of lists containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
# OUTPUT: confusion_matrix : the confusion matrix computed based on the classified_data. The order of elements needs
#                            to be the same as  in the classification scheme. The columns correspond to actual classes
#                            and rows to predicted classes. In other words, confusion_matrix[0] should be understood
#                            as the row of values predicted as Female, and [row[0] for row in confusion_matrix] as the
#                            column of values that were actually Female

def confusionMatrix(classified_data):
    # Initialize the set of classes present in the data
    classes = set([row[1] for row in classified_data])

    # Create a dictionary to map classes to indices in the confusion matrix
    class_dict = {}
    for i, cls in enumerate(sorted(classes)):
        class_dict[cls] = i

    # Initialize the confusion matrix with the correct size and values
    num_classes = len(classes)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for row in classified_data[2:]:
        actual_class = row[1]
        predicted_class = row[2]
        if predicted_class == '':
            continue
        actual_index = class_dict[actual_class]
        predicted_index = class_dict[predicted_class]
        confusion_matrix[actual_index][predicted_index] += 1

    return confusion_matrix



# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# INPUT: confusion_matrix : the confusion matrix computed based on the classified_data. The order of elements is
#                           the same as  in the classification scheme. The columns correspond to actual classes
#                           and rows to predicted classes.
# OUTPUT: a list of appropriate true positive, false positive or false
#         negative values per a given class, in the same order as in the classification scheme. For example, tps[1]
#         corresponds for TPs for Male class.


def computeTPs(confusion_matrix):
    tps = [confusion_matrix[i][i] for i in range(len(confusion_matrix))]
    return tps


def computeFPs(confusion_matrix):
    fps = []
    for i in range(len(confusion_matrix)):
        fp = sum(confusion_matrix[j][i] for j in range(len(confusion_matrix)) if j != i)
        fps.append(fp)
    return fps


def computeFNs(confusion_matrix):
    fns = []
    for i in range(len(confusion_matrix)):
        fn = sum(confusion_matrix[i][j] for j in range(len(confusion_matrix)) if j != i)
        fns.append(fn)
    return fns


# These functions compute the evaluation measures based on the provided values. Not all measures use of all the values.
#
# INPUT: tps, fps, fns, data_size
#                       : the per-class true positives, false positive and negatives, and size of the classified data.
# OUTPUT: appropriate evaluation measures created using the macro-average approach.

def computeMacroPrecision(tps, fps, fns, data_size):
    tp_sum = sum(tps)
    fp_sum = sum(fps)
    if tp_sum == 0 and fp_sum == 0:
        return 0
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    return precision


def computeMacroRecall(tps, fps, fns, data_size):
    tp_sum = sum(tps)
    fn_sum = sum(fns)
    if tp_sum == 0 and fn_sum == 0:
        return 0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    return recall


def computeMacroFMeasure(tps, fps, fns, data_size):
    precision = computeMacroPrecision(tps, fps, fns, data_size)
    recall = computeMacroRecall(tps, fps, fns, data_size)
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f_measure


def computeAccuracy(tps, fps, fns, data_size):
    tn = data_size - sum(tps) - sum(fps) - sum(fns)
    accuracy = (sum(tps) + tn) / data_size if data_size > 0 else 0
    return accuracy


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.

# INPUT: classified_data   : a list of lists containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
#       confusion_func     : function to be invoked to compute the confusion matrix
#
# OUTPUT: computed measures
def evaluateKNN(classified_data, confusion_func=confusionMatrix):
    # Initialize the set of classes present in the data
    classes = set([row[1] for row in classified_data])

    # Create a dictionary to map classes to indices in the confusion matrix
    class_dict = {}
    for i, cls in enumerate(sorted(classes)):
        class_dict[cls] = i

    # Initialize the confusion matrix with the correct size and values
    num_classes = len(classes)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for row in classified_data[2:]:
        actual_class = row[1]
        predicted_class = row[2]
        if predicted_class == '':
            continue
        actual_index = class_dict[actual_class]
        predicted_index = class_dict[predicted_class]
        confusion_matrix[predicted_index][actual_index] += 1

    # Compute evaluation metrics using the confusion matrix
    tps = computeTPs(confusion_matrix)
    fps = computeFPs(confusion_matrix)
    fns = computeFNs(confusion_matrix)
    data_size = len(classified_data) - 2
    precision = computeMacroPrecision(tps, fps, fns, data_size)
    recall = computeMacroRecall(tps, fps, fns, data_size)
    f_measure = computeMacroFMeasure(tps, fps, fns, data_size)
    accuracy = computeAccuracy(tps, fps, fns, data_size)

    return precision, recall, f_measure, accuracy


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["classified_data"]}')
    classified_data = Task_1_5.readCSVFile(opts['classified_data'])
    print('Evaluating kNN')
    result = evaluateKNN(classified_data, eval(opts['cf']))
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt


# def classification(p, r, f, s, avg):
#     d = {'precision': p,
#          'recall': r,
#          'f1-score': f,
#          'support': s}
#     frame = pd.DataFrame(data=raw_data)
#
#     print("----------------------------------------------------")
#     print("|               Classification Report              |")
#     print("----------------------------------------------------")
#     print(frame)
#     print("")
#     print('avg/total    {}    {}    {}    {}'.format(avg[0], avg[1], avg[2], avg[3]))
#
#
# # raw_data = {'precision': [0.84, 0.87],
# #             'recall': [0.88, 0.83],
# #             'f1-score': [0.86, 0.85],
# #             'support': [150, 150]}
# # df = pd.DataFrame(data=raw_data)
# #
# # print("----------------------------------------------------")
# # print("|               Classification Report              |")
# # print("----------------------------------------------------")
# # print(df)
# # print("")
# # print('avg/total    0.86    0.86    0.86    300')
#
#
# def confusion(a, b):
#     print("----------------------------------------------------")
#     print("|                  Confusion Matrix                |")
#     print("----------------------------------------------------")
#     print('[{}'.format(a))
#     print(' {}]'.format(b))
#     print("")
#
#
# # print("----------------------------------------------------")
# # print("|                  Confusion Matrix                |")
# # print("----------------------------------------------------")
# # print('[{}'.format('[1 0]'))
# # print(' {}]'.format('[1 0]'))
# # print("")
#
# # cm_list = metrics.confusion_matrix(expected, predicted).tolist()
# # list_total = float(sum(sum(x) for x in cm_list))
# #
#
# def pos_neg(pos, neg):
#     print("----------------------------------------------------")
#     print("|           False Positives and Negatives          |")
#     print("----------------------------------------------------")
#     print("False Positive: ", pos)
#     print("")
#     print("False Negative: ", neg)
#     print("")
#
#
# # print("----------------------------------------------------")
# # print("|           False Positives and Negatives          |")
# # print("----------------------------------------------------")
# # print("False Positive: ", 0.5)  # 0.08333333333333333
# # print("")
# # print("False Negative: ", 0.0)  # 0.06
# # print("")
#
#
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# def plot_it(ar):  # [[1, 2], [3, 4]]
#     plt.figure()
#     arr = np.array(ar)
#     lab = np.array(['Spammer', 'Non-Spammer'])
#     plot_confusion_matrix(arr, lab)
#     plt.show()
#
#
# plot_it([[1, 0], [1, 0]])


print('Hello, world!')
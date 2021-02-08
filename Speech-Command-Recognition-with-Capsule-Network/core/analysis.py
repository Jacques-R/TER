from sklearn.metrics import confusion_matrix
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from termcolor import cprint

label_30 = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 
        'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

label_11 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'unrecognized']

class Analysis(object):
    def __init__(self,args):
        self.args = args
        self.SavePath_Generate()
        
    def SavePath_Generate(self):
        ex_name = self.args.ex_name+'_'+self.args.train_with
        ex_name += '_open_set' if self.args.open_set else ''
        save_path = os.path.join(self.args.project_path,'save',self.args.model,ex_name)
        self.SavePath = save_path
        return None

    def CreateCsv(self, AnalysisPath):
        AnalysisFile = open(AnalysisPath,'w')
        return AnalysisFile

    def Matrix2Csv(self, Array, filename):
        '''
        save 2-dimensional array as a csv file.
        [input] Array: array want to be saved as csv file. Should be 2-dim
                path: where the csv file will saved.
                name(str): name of csv file. 'noname' as default.
        [output] None.
        '''
        save_path = os.path.join(self.SavePath, 'MatrixCSV')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        AnalysisPath = save_path + filename
        AnalysisFile = self.CreateCsv(AnalysisPath=AnalysisPath)
        assert len(Array.shape) == 2
        for i in range(Array.shape[0]):
            for j in range(Array.shape[1]):
                AnalysisFile.write(str(Array[i,j])+',')
            AnalysisFile.write('\n')
        AnalysisFile.close()
        return None

    def Matrix2Png(self, filename):
        plt.savefig(self.SavePath + filename)

    def ConfusionMatrix(self,y_pred,teY, labels):
        '''
        input: y_pred & teY #shape:(N,)
        output: confusion matrix(2d) 
        # the number of observations known to be in group i but predicted to be in group j.
        # Row: Predict / Column: True
              Predict
             ---------
        True |  Value
        '''
        matrix = confusion_matrix(teY, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        matrix = np.around(matrix, 2)
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot()
        return matrix

    def RawPredictMatrix_Generate(self,y_pred30,teY30):
        self.Matrix2Csv(Array=y_pred30,filename='/Predict30_'+str(self.args.model)+'.csv')
        self.Matrix2Csv(Array=teY30,filename='/GroundTruth30_'+str(self.args.model)+'.csv')
        return None

    def MatrixSave(self,y_pred30,teY30):
        np.save(self.SavePath + '/Predict30_'+str(self.args.model)+'.npy', y_pred30)
        np.save(self.SavePath + '/GroundTruth30_'+str(self.args.model)+'.npy', teY30)

#######################################################################################
################################# FROM SKLEARN 0.23.2 #################################
########################### Then modified by Sacha Carniere ###########################
#######################################################################################

from itertools import product

import numpy as np


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    display_labels : ndarray of shape (n_classes,), default=None
        Display labels for plot. If None, display labels are set from 0 to
        `n_classes - 1`.

    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.

    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.

    ax_ : matplotlib Axes
        Axes with confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.

        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(18,18))
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        max_value = cm.max()
        second_max_value = np.sort(cm.max(axis=0))[-2]
        vmax = second_max_value+0.1*max_value
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmax=vmax)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (vmax + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


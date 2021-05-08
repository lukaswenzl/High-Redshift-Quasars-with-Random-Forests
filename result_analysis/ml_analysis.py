import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
from scipy.stats import sigmaclip
# from sklearn.learning_curve import learning_curve
# from sklearn.learning_curve import validation_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def evaluate_regression(y_test,y_pred):
    """
    This routine calculates the explained variance score, the mean absolute
    error and the mean squared error and prints them.

    Parameters:
        y_test : array-like, shape (n_samples)
        Array containing the true values of the regression

        y_pred : array-like, shape (n_samples)
        Array containing the predicted values of the regression
    """

    print("Explained Variance Score")
    print(explained_variance_score(y_test, y_pred, multioutput='raw_values'))
    print("\n")
    print("Mean Absolute Error")
    print(mean_absolute_error(y_test,y_pred, multioutput='raw_values'))
    print("\n")
    print("Mean Squared Error")
    print(mean_squared_error(y_test,y_pred, multioutput='raw_values'))


def show_features(df, features, labels, frac=1.0):
    """ This plot will display a fraction (frac) of all objects in the
    DataFrame (df) in 2D-feature spaces. Not all feature combinations are
    displayed but only the ones following after another in the list (features).
    The objects will be displayed in colors corresponding to their labels in
    the DataFrame. We use level 5.0 sigma clipping to center the axes on the
    majority of the objects.

    Input:
            df (DataFrame) DataFrame containing features and labels
            features (list) list of features in the DataFrame
            labels (list) list of the names of the labels in df.label
            frac (float) fraction of objects in df to be displayed

    Output:
            matplotlib figure object
    """

    df = df.sample(frac=frac)

    cols = 3
    gs = gridspec.GridSpec(len(features) // cols , cols)


    fig = plt.figure(figsize=(9,3*(len(features) // cols)),dpi=100)
    ax = []

    for i in range(len(features)-1):
        row = (i // cols)
        col = i % cols
        ax.append(fig.add_subplot(gs[row, col]))

        color=iter(cm.rainbow(np.linspace(0,1,len(labels))))

        #sigma clipping here
        x_range,xlow,xup = sigmaclip(df[features[i]], low=5.0, high=5.0)

        y_range,ylow,yup = sigmaclip(df[features[i+1]], low=5.0, high=5.0)

        for j in range(len(labels)):
            dfj = df.query('label =="'+str(labels[j])+'"')
            ax[-1].scatter(dfj[features[i]], dfj[features[i+1]], \
            alpha=0.2, c=next(color),edgecolor='None' )

        ax[-1].set_xlim(xlow,xup)
        ax[-1].set_ylim(ylow,yup)
        ax[-1].set_xlabel(str(features[i]))
        ax[-1].set_ylabel(str(features[i+1]))

    plt.tight_layout()

    plt.show()


# def plot_learning_curve(estimator, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     ADAPTED FROM SCIKIT LEARN EXAMPLES:
#     http://scikit-learn.org/stable/auto_examples/model_selection/
#     plot_learning_curve.html
#
#     Generate a simple plot of the test and training learning curve.
#
#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.
#
#     title : string
#         Title for the chart.
#
#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.
#
#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.
#
#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.
#
#     cv : integer, cross-validation generator, optional
#         If an integer is passed, it is the number of folds (defaults to 3).
#         Specific cross-validation objects can be passed, see
#         sklearn.cross_validation module for the list of possible objects
#
#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     # Tex font
#     rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#     rc('text', usetex=True)
#
#     fig = plt.figure(num=None,figsize=(6,6), dpi=140)
#     fig.subplots_adjust(left=0.15, bottom=0.1, right=0.98, top=0.96)
#     ax = fig.add_subplot(1,1,1)
#
#
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel(r'$\rm{Training\ examples}$', fontsize=20)
#     plt.ylabel(r'$\rm{Score}$', fontsize=20)
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     min_train = min(train_scores_mean)
#     min_test = min(test_scores_mean)
#     min_all = min(min_test,min_train)
#     max_std = max(max(train_scores_std),max(test_scores_std))
#
#     ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label=r'$\rm{Training\ score}$')
#     ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label=r'$\rm{Cross-validation\ score}$')
#
#     plt.legend(loc="best")
#
#     ax.set_ylim(min_all-max_std*3,1.0)
#
#     return plt
#
# def plot_validation_curve(estimator, param_name, param_range, title, X, y,
#                                                 ylim=None, cv=None, n_jobs=1):
#
#     """ This plot will calculate the validation curve for the chosen estimator,
#     hyper-parameter and range of that hyper-parameter. It will return a plt
#     object.
#
#     Input:
#             estimator : object type that implements the "fit" and "predict" methods
#                 An object of that type which is cloned for each validation.
#
#             param_name : string
#                 A string with the parameter name fit for the estimator
#
#             param_range : array-like
#                 A list of parameter values
#
#             title : string
#                 Title string
#
#             X : array-like, shape (n_samples, n_features)
#                 Training vector
#
#             y : array-like, shape (n_samples) or (n_samples, n_features)
#                 The classification vector
#
#             tuple, shape (ymin, ymax), optional
#                 Defines minimum and maximum yvalues plotted.
#
#             cv : integer, cross-validation generator, optional
#                 If an integer is passed, it is the number of folds (defaults to 3).
#                 Specific cross-validation objects can be passed, see
#                 sklearn.cross_validation module for the list of possible objects
#
#             n_jobs : integer, optional
#                 Number of jobs to run in parallel (default 1).
#
#     Output:
#             matplotlib figure object
#     """
#
#     print "THIS FUNCTION IS DEPRECATED"
#
#     plt.figure()
#     plt.title(title)
#
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#
#     train_scores, test_scores = validation_curve(
#         estimator, X, y, param_name=param_name, param_range=param_range,
#         cv=10, scoring="accuracy", n_jobs=1)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#
#     plt.title(title)
#     plt.xlabel(param_name)
#     plt.ylabel("Score")
#
#     plt.plot(param_range, train_scores_mean, label="Training score", color="r")
#     plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.2, color="r")
#     plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#                  color="g")
#     plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.2, color="g")
#     plt.legend(loc="best")
#
#     return plt


def plot_precision_recall_curve(y_true, y_prob_pred, pos_label):

    # Tex font
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    fig = plt.figure(num=None,figsize=(6,6), dpi=140)
    fig.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.96)
    ax = fig.add_subplot(1,1,1)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob_pred,
    pos_label=pos_label)

    # average_precision = average_precision_score(y_true, y_prob_pred,
                                        # pos_label=pos_label,average="micro")

    average_precision = 0


    # Plot Precision-Recall curve
    ax.plot(recall, precision, 'b', linewidth=2)
    ax.set_xlabel(r'$\rm{Recall}$', fontsize=20)
    ax.set_ylabel(r'$\rm{Precision}$', fontsize=20)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    ax.grid(True)
    # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, pos_label=None):
    """
    This function calculates the receiver operating curve and returns a
    matplotlib plot of the curve.

    Parameters:
        y_true : array-like, shape (n_samples)
        Array containing the true values of the regression

        y_pred_proba : array-like, shape (n_samples)
        Array containing the predicted probabilities of the regression

        pos_label : string
        String of the positive label to use for the ROC curve

    Returns:
        plt : matplotlib plot

    """

    # Tex font
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    # Calculation of the false positive rate (fpr) and the true positive rate (tpr)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, y_pred_proba, pos_label ="QSO")

    # Evaluating the are under the curve
    auc_score = auc(fpr_rf,tpr_rf)

    fig = plt.figure(num=None,figsize=(6,6), dpi=140)
    fig.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.96)
    ax = fig.add_subplot(1,1,1)


    ax.plot([0, 1], [0, 1], 'k--',linewidth=1.5)
    plt.plot(fpr_rf, tpr_rf, 'b', linewidth=2)

    ax.text(0.8,0.05,r'$\rm{AUC} : $'+r'${0:.2f}$'.format(auc_score)
          ,fontsize=15,transform = ax.transAxes)

    ax.set_xlabel(r'$\rm{False\ positive\ rate}$', fontsize=20)
    ax.set_ylabel(r'$\rm{True\ positive\ rate}$', fontsize=20)

    ax.set_ylim(0.0,1.05)
    ax.set_xlim(-0.05,1.00)

    return plt


def plot_confusion_matrix(cnf_matrix, classes,
                          normalize=False,
                          title=r'$\rm{Confusion\ matrix}$',
                          cmap=plt.cm.Blues):
    """
    ADAPTED FROM SCIKIT LEARN EXAMPLES:
    http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    This function plots the confusion matrix of a classification as an image
    using imshow.

    Parameters:

        cnf_matrix: array-like
        The confusion matrix as calculated by sklearn.metrics.confusion_matrix

        classes : list of strings
        This is the list of class names. It can be provided by clf.classes_

        normalize: boolean
        Boolean to chose normalization of each row to 1

        title: string
        The title of the confusion matrix

        cmap: colormap
        The color map to use for coloring the matrix elements

    Returns:
        plt : matplotlib plot

    """

    # Tex font
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('text', usetex=True)

    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        cnf_matrix = np.around(cnf_matrix,decimals=3)
        print(r'$\rm{Normalized\ confusion\ matrix}$')
    else:
        print(r'$\rm{Confusion\ matrix}$')

    fig = plt.figure(num=None,figsize=(6,6), dpi=100)
    fig.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.96)
    ax = fig.add_subplot(1,1,1)

    im = ax.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    cbar = fig.colorbar(im,ax=ax)
    # label=r'$\rm{Density}$'
    # cbar.set_label(label,fontsize=14)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    class_names = []
    for cl in classes:
        class_names.append(r'$\rm{'+str(cl)+'}$')

    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names, rotation=45)


    ax.set_ylabel(r'$\rm{True\ label}$',fontsize=15)
    ax.set_xlabel(r'$\rm{Predicted\ label}$',fontsize=15)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(list(range(cnf_matrix.shape[0])), \
                                        list(range(cnf_matrix.shape[1]))):
        plt.text(j, i,r'$\rm{'+str(cnf_matrix[i, j])+'}$',
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")




    return plt


def my_confusion_matrix(cnf_matrix, classes,
                          title=r'$\rm{Confusion\ matrix}$',
                          cmap=plt.cm.Blues):
    """
    ADAPTED FROM SCIKIT LEARN EXAMPLES:
    http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    This function plots the confusion matrix of a classification as an image
    using imshow.

    Parameters:

        cnf_matrix: array-like
        The confusion matrix as calculated by sklearn.metrics.confusion_matrix

        classes : list of strings
        This is the list of class names. It can be provided by clf.classes_

        normalize: boolean
        Boolean to chose normalization of each row to 1

        title: string
        The title of the confusion matrix

        cmap: colormap
        The color map to use for coloring the matrix elements

    Returns:
        plt : matplotlib plot

    """


    # Tex font
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('text', usetex=True)


    n_cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    n_cnf_matrix = np.around(n_cnf_matrix,decimals=3)

    fig = plt.figure(num=None,figsize=(6,6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.96)
    ax = fig.add_subplot(1,1,1)

    im = ax.imshow(n_cnf_matrix, interpolation='nearest', cmap=cmap)
    #ax.set_title(title)
    #cbar = fig.colorbar(im,ax=ax)
    #label=r'$\rm{Fraction}$'
    #cbar.set_label(label,fontsize=14)
    classes = classes[:] ####
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    class_names = []
    for cl in classes:
        class_names.append(r'$\rm{'+str(cl)+'}$')

    ax.set_xticklabels(class_names, rotation=45, fontsize = 12)
    ax.set_yticklabels(class_names, rotation=45, fontsize = 12)


    ax.set_ylabel(r'$\rm{True\ label}$',fontsize=24)
    ax.set_xlabel(r'$\rm{Predicted\ label}$',fontsize=24)

    thresh = n_cnf_matrix.max() *0.6 ###what does this do
    for i, j in itertools.product(list(range(cnf_matrix.shape[0])), \
                                        list(range(cnf_matrix.shape[1]))):

        if n_cnf_matrix[i,j] > 0:
            plt.text(j, i-0.175,r'$\rm{'+str(cnf_matrix[i, j])+'}$',
                    va='center',horizontalalignment='center', size =12,
                     color="white" if n_cnf_matrix[i, j] > thresh else "black")

            #plt.text(j, i+0.2,r'$\rm{'+str(n_cnf_matrix[i, j]*100))+'}\%$',
            #         va='center',horizontalalignment='center', size =12,
            #         color="white" if n_cnf_matrix[i, j] > thresh else "black")
            plt.text(j, i+0.2,'{:.1f}%'.format(n_cnf_matrix[i, j]*100),
                     va='center',horizontalalignment='center', size =12,
                     color="white" if n_cnf_matrix[i, j] > thresh else "black")

        elif cnf_matrix[i,j ]>0:
            plt.text(j, i-0.175,r'$\rm{'+str(cnf_matrix[i, j])+'}$',
                    va='center',horizontalalignment='center', size =12,
                     color="white" if n_cnf_matrix[i, j] > thresh else "black")
        else:
            plt.text(j, i,'-',
            va='center',horizontalalignment='center', size =12,
            color="white" if n_cnf_matrix[i, j] > thresh else "black")




    return plt

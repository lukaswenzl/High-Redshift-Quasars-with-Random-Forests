"""Helper functions for random forest classification and regression
Author Lukas Wenzl"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor


#from . import ml_sets as sets
from result_analysis import ml_analysis as ml_an
from result_analysis import photoz_analysis as pz_an

#to standardize scaling
from sklearn.preprocessing import RobustScaler

#clean up memory
import gc
import math


def build_matrices(df, features,label, drop_nans = True):

    """This routines returns the feature matrix X to use for the classification
    and the label vector y based on the input DataFrame. The label column must
    be df.label and the features must be valid column names of the DataFrame

    Input:
            df (DataFrame)
            features (list) list of label names to be considered
    Output:
            X (Numpy Array, 2D) feature matrix
            y (Numpy Array, 1D) label vector
    """

    if drop_nans:
        df.dropna(axis=0,how='any',subset=features,inplace=True)

    X = np.array(df[features])
    y = np.array(df[label])

    return X,y

def build_matrix(df, features,drop_nans = False):

    """This routines returns the feature matrix X to use for the classification.
    The features must be valid column names of the DataFrame.

    Input:
            df (DataFrame)
            features (list) list of label names to be considered
    Output:
            X (Numpy Array, 2D) feature matrix
    """

    if drop_nans:
        df.dropna(axis=0,how='any',subset=features,inplace=True)

    X = np.array(df[features])

    return X


def rf_class_grid_search(df_train,df_pred, features, label, param_grid, rand_state, scores, name):
    """This routine calculates the random forest classification on a grid of
    hyper-parameters for the random forest method to test the best
    hyper-parameters. The analysis results of the test will be written out and
    saved.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            param_grid : dictionary-like structure
            Parameter grid of input parameters for the grid search

            rand_state : integer
            Setting the random state variables to ensure reproducibility

            scores : list of strings
            Setting the score by which the grid search should be evaluated

            name : strings
            Setting the name of the output file for the grid search which
            contains all information about the grid

    """


    X_train, y_train = build_matrices(df_train, features,label=label)
    X_test, y_test = build_matrices(df_pred, features,label=label)

    print ("Trainingset: ", X_train.shape)
    print(pd.Series(y_train).value_counts())

    print("Testset:",  X_test.shape)
    print(pd.Series(y_test).value_counts())

    for score in scores:
        print(("# Tuning hyper-parameters for %s" % score))
        print()

        clf = GridSearchCV(RandomForestClassifier(random_state=rand_state),
            param_grid, cv=5, scoring='%s' % score, n_jobs = 15, return_train_score=True)

        clf.fit(X_train, y_train)

        print("Detailed classification report:")
        print("")
        print("The model is trained on the training set.")
        print("The scores are computed on the test set.")
        print("")
        y_true, y_pred = y_test, clf.predict(X_test)
        y_true = y_true.astype('str')
        y_pred = y_pred.astype('str')

        print((classification_report(y_true, y_pred)))
        print()

        print("Best parameters set found on training set:")
        print()
        print((clf.best_params_))
        print()
        print("Grid scores on training set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print(("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params)))
        print()
        df = pd.DataFrame(clf.cv_results_)
        df.to_hdf('data/'+name+'_'+score+'.hdf5','data')



def rf_class_validation_curve(df, features, label, params, param_name, param_range):
    """This routine calculates the validation curve for one hyper-parameter of
    the random forest classification method.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame

            label : string
            The label for the regression

            param_name (string) name of the hyper parameter
            param_range (list) list of parameter values to use


    Output:
            None
    """

    print("THIS FUNCTION IS DEPRECATED")

    X,y = build_matrices(df, features,label)

    # Standardizing the data
    # X = preprocessing.robust_scale(X)

    clf = RandomForestClassifier(**params)
    title = "Validation curve / Random Forest Classifier"
    ml_an.plot_validation_curve(clf, param_name, param_range, title, X, y,
                                            ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()


def  rf_class_create(df_train, features, label, params,rand_state, save=False, save_filename=None):
    """This routine creates a random forest classifier.
     It is aimed at multi-class classification and used for my pipeline.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            rand_state : integer
            Setting the random state variables to ensure reproducibility

    Return :
            clf : scikit-learn Classifier
            The Classifier trained on the training set

    """
    print("training on "+ str(len(df_train.index))+" entries")


    X_train, y_train = build_matrices(df_train, features,label=label)

    # Standardizing the data
    # X_train = preprocessing.robust_scale(X_train)
    # X_pred = preprocessing.robust_scale(X_pred)
    clf = RandomForestClassifier(**params)

    clf.fit(X_train,y_train)

    if(save):
        from sklearn.externals import joblib
        joblib.dump(clf, save_filename+'.pkl')

    feat_importances = clf.feature_importances_

    print("Feature Importance ")
    for i in range(len(features)):
        print(str(features[i])+": "+str(feat_importances[i]))
    print("\n")

    return clf


def  rf_class_predict(clf, df_pred, features, prefix):
    """This routine takes a random forest classifier and applies it to data
     It is aimed at multi-class classification and used for my pipeline.


    Parameters:
            clf : RandomForestClassifier
            Classifier from sklearn.

            df_pred : pandas dataframe
            Contains the data to classify

            features : list of strings
            List of features

            prefix : string
            Prefix to all new columns to be able to use multiple
            classifier.

    Return :
            df_pred : pandas dataframe
            The dataframe containing the features for prediction (given as argument to the function)
            and the classification in the pred_label named column.
    """

    X_pred = build_matrix(df_pred, features)

    # Standardizing the data
    # X_train = preprocessing.robust_scale(X_train)
    # X_pred = preprocessing.robust_scale(X_pred)

    y_pred = clf.predict(X_pred)

    # Predicting the probabilities for the classes
    y_prob = clf.predict_proba(X_pred)

    df_prob = pd.DataFrame(y_prob)
    df_prob.columns = clf.classes_
    df_prob.index = df_pred.index #not sure how this works
    df_prob['qso_prob'] = df_prob.highz + df_prob.midz + df_prob.lowz + df_prob.vlowz

    #df_prob['qso_prob'] = 0
    #for i in [x for x in df_prob.columns if "z" in x]:
    #    df_prob['qso_prob'] = df_prob['qso_prob'] + df_prob[i]

    df_prob['pred_class'] = y_pred

    #add prefix for the classifier (if using multiple)
    df_prob = df_prob.add_prefix(prefix)

    df_pred = pd.concat([df_pred, df_prob], axis=1)

    del df_prob,X_pred,y_pred,y_prob
    gc.collect()

    return df_pred



def rf_class_example(df_train, df_pred, features, label, params, rand_state, save=False, save_filename=None, display=True):
    """This routine calculates an example of the random forest classification
     method. It is aimed at multi-class classification.
     It prints the classification report and feature importances and shows the
     confusion matrix for all classes.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            rand_state : integer
            Setting the random state variables to ensure reproducibility
    """


    #clf, y_pred, y_prob = rf_class_predict(df_train,df_pred, features, label,
    #                                                        params, rand_state)

    X_train, y_train = build_matrices(df_train, features,label=label)


    clf = RandomForestClassifier(**params)

    clf.fit(X_train,y_train)

    X_pred, y_true = build_matrices(df_pred, features,label=label)

    # y_true = y_true.astype('string')
    # y_pred = y_pred.astype('string')
    y_pred = clf.predict(X_pred)
    y_prob = clf.predict_proba(X_pred)

    df_prob = pd.DataFrame(y_prob)
    df_prob.columns = clf.classes_
    df_prob.index = df_pred.index
    #df_prob['qso_prob'] = df_prob.highz + df_prob.midz + df_prob.lowz + df_prob.vlowz

    df_prob['qso_prob'] = 0
    for i in [x for x in df_prob.columns if "z" in x]:
        df_prob['qso_prob'] = df_prob['qso_prob'] + df_prob[i]


    df_prob['true_class'] = y_true
    df_prob['pred_class'] = y_pred

    if(save):
        from sklearn.externals import joblib
        joblib.dump(clf, save_filename+'.pkl')

    feat_importances = clf.feature_importances_

    print("Classification Report ")
    print((classification_report(y_true, y_pred)))
    print("\n")
    print("Feature Importance ")
    for i in range(len(features)):
        print(str(features[i])+": "+str(feat_importances[i]))
    print("\n")

    # Confusion matrix
    if (display):
        all_class_names = clf.classes_
        class_names = ["A", "F", "G", "K", "M", "L", "T", "highz", "midz", "lowz", "vlowz"]
        #for name
        cnf_matrix = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)


        # ml_an.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        #   title='Confusion matrix, with normalization')

        # ml_an.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                        #   title='Confusion matrix, without normalization')

        print(class_names)
        ml_an.my_confusion_matrix(cnf_matrix, classes=class_names)

        plt.show()




    return y_true, y_pred, df_prob

def rf_reg_grid_search(df,features,label,param_grid,rand_state,scores,name):
    """This routine calculates the random forest regression on a grid of
    hyper-parameters for the random forest method to test the best
    hyper-parameters. The analysis results of the test will be written out and
    saved.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            param_grid : dictionary-like structure
            Parameter grid of input parameters for the grid search

            rand_state : integer
            Setting the random state variables to ensure reproducibility

            scores : list of strings
            Setting the score by which the grid search should be evaluated

            name : strings
            Setting the name of the output file for the grid search which
            contains all information about the grid

    """

    X,y = build_matrices(df, features,label)

    # Standardizing the data
    X = preprocessing.robust_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2,random_state=rand_state)

    print("Training sample size: ", X_train.shape)
    print("Evaluation sample size: ", X_test.shape)

    for score in scores:
        print(("# Tuning hyper-parameters for %s" % score))
        print()

        reg = GridSearchCV(RandomForestRegressor(random_state=rand_state), \
                        param_grid,scoring='%s' % score,cv=5,n_jobs=15, return_train_score=True)

        reg.fit(X_train, y_train)

        print("Best parameters set found on training set:")
        print()
        print((reg.best_params_))
        print()
        print("Grid scores on training set:")
        print()
        means = reg.cv_results_['mean_test_score']
        stds = reg.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, reg.cv_results_['params']):
            print(("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params)))
        print()
        df = pd.DataFrame(reg.cv_results_)
        #df.to_hdf('RF_GS_'+name+'_'+score+'.hdf5','data')
        print()
        print("The model is trained on the full development set (80%).")
        print("The scores are computed on the full evaluation set (20%).")
        print()
        y_true, y_pred = y_test, reg.predict(X_test)
        ml_an.evaluate_regression(y_test,y_pred)
        pz_an.evaluate_photoz(y_test,y_pred)
        print()




def rf_reg_validation_curve(df,features,label,params,val_param,val_range):
    """This routine calculates the validation curve for random forest
    regression.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            val_param : string
            Name of the validation parameter

            val_range : array-like
            List of parameter values for the validation curve

    """

    print("THIS FUNCTION IS DEPRECATED")

    X,y = build_matrices(df, features,label)

    # Random Forest Regression
    reg = RandomForestRegressor(**params)

    #Calculate and plot validation curve
    pz_an.plot_validation_curve(reg, val_param, val_range, X, y,
                                        ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()


def rf_reg_predict(reg, scaler, df, features, pred_label):
    """This function predicts the regression values for pred_set based on the
    features specified in the train_set

    Parameters:
            reg : trained regressor
            The regressor trained on the data

            df : pandas dataframe
            The dataframe containing the features for prediction

            features : list of strings
            List of features

            pred_label : string
            Name of the new label in the df dataframe in which the
            predicted values are written

    Output:
            df : pandas dataframe
            The dataframe containing the features for prediction and the
            regression values in the pred_label named column.
    """
    #df = df.copy()
    # Building test and training sample
    X = build_matrix(df, features)

    # Standardizing the data
    X = scaler.transform(X)

    #predicting the redshift
    redshift = reg.predict(X)
    #redshift= pd.DataFrame(redshift)

    #redshift.index = df.index #not sure how this works

    #df.loc[:,pred_label]=redshift
    df[pred_label]=redshift
    #df = df.assign(pred_label=redshift)

    del X
    gc.collect()

    return df

def rf_create_reg(df,features,label,params,rand_state,save=False,save_filename=None):
    """This routinecreates the random forest regression tuned
    to photometric redshift estimation. This is the method used by the pipeling to
    create the regressor

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            rand_state : integer
            Setting the random state variables to ensure reproducibility

            save : Boolean
            specifies if the result should be saved

            save_filename : string
            The Filename as which the regressor should be saved
            The scaler is also saved, with the appendix _scaler

    """

    # Building test and training sample
    X_train,y_train = build_matrices(df, features, label)

    # Standardizing the data
    scaler = RobustScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    # Save scale
    if(save):
        if save_filename:
            from sklearn.externals import joblib
            joblib.dump(scaler, save_filename+'_scaler.pkl')
        else:
            print("Error: No Filename supplied!")




    #X_train, X_test, y_train, y_test = train_test_split(
    #    X,y, test_size=0.2, random_state=rand_state)

    # Random Forest Regression
    reg = RandomForestRegressor(**params)
    reg.fit(X_train,y_train)
    #y_pred = reg.predict(X_test)



    # Save regressor
    if(save):
        if save_filename:
            from sklearn.externals import joblib
            joblib.dump(reg, save_filename+'.pkl')
        else:
            print("Error: No Filename supplied!")

    feat_importances = reg.feature_importances_

    # Evaluate regression method
    print("Feature Importances ")
    for i in range(len(features)):
        print(str(features[i])+": "+str(feat_importances[i]))
    print("\n")



def rf_reg_example(df,features,label,params,rand_state,save=False,save_filename=None, display=True):
    """This routine calculates an example of the random forest regression tuned
    to photometric redshift estimation. The results will be analyzed with the
    analyis routines/functions provided in ml_eval.py and photoz_analysis.py

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            rand_state : integer
            Setting the random state variables to ensure reproducibility

    returns the fitted regressor

    """

    # Building test and training sample
    X,y = build_matrices(df, features, label)

    # Standardizing the data
    scaler = RobustScaler().fit(X)
    X = scaler.transform(X)
    # Save scale
    if(save):
        if save_filename:
            from sklearn.externals import joblib
            joblib.dump(scaler, save_filename+'_scaler.pkl')
        else:
            print("Error: No Filename supplied!")




    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=rand_state)

    # Leftover from trying out weights
    # w_train = X_train[:,-1]
    # X_train = X_train[:,:-1]
    # w_test = X_test[:,-1]
    # X_test = X_test[:,:-1]

    # Random Forest Regression
    reg = RandomForestRegressor(**params)

    reg.fit(X_train,y_train)

    y_pred = reg.predict(X_test)

    feat_importances = reg.feature_importances_


    # Save regressor
    if(save):
        if save_filename:
            from sklearn.externals import joblib
            joblib.dump(reg, save_filename+'.pkl')
        else:
            print("Error: No Filename supplied!")




    # Evaluate regression method
    print("Feature Importances ")
    for i in range(len(features)):
        print(str(features[i])+": "+str(feat_importances[i]))
    print("\n")

    if(display):
        ml_an.evaluate_regression(y_test,y_pred)


        pz_an.plot_redshifts(y_test,y_pred)
        pz_an.plot_error_hist(y_test,y_pred)
        plt.show()


        pz_an.plot_error_hist(y_test[np.where(y_test > 4.7)],y_pred[np.where(y_test > 4.7)])
        plt.title('error histogram only for quasars with z bigger 4.7')
        plt.show()

        pz_an.plot_error_hist(y_test[np.where(y_test > 5.4)],y_pred[np.where(y_test > 5.4)])
        plt.title('error histogram only for quasars with z bigger 5.4')
        plt.show()
    return reg, scaler



def make_train_pred_set(df_stars, df_qsos, test_ratio ,rand_state,
                    save_prefix = 'default', concat=True, save = False):
    """ This routine combines the already labelled quasar and star flurx ratio
    catalogs and creates a training and test set from them with the
    train_test_split function of scikit-learn.

    Parameters:
            df_star : pandas dataframe
            Star flux ratio catalog

            df_qsos : pandas dataframe
            Quasar flux ratio catalog

            test_ratio : float
            Ratio of the test set with respect to the total combined catalogs.
            The value ranges between 0.0 and 1.0.

            rand_state: integer
            Integer that sets the random state variable for reproducibility

            save : boolean
            Boolean to select whether the test and training sets are saved

            concat : boolean
            Boolean to select whether the samples are already concatenated or
            returned without.

    Returns:
            df_train : pandas dataframe
            The new combined training set

            df_test : pandas dataframe
            The new combined test set
    """

    stars_train, stars_test = train_test_split(df_stars, test_size=test_ratio,
                                                    random_state=rand_state)

    qsos_train, qsos_test = train_test_split(df_qsos, test_size=test_ratio,
                                                    random_state=rand_state)

    df_train = pd.concat([stars_train,qsos_train], sort=False)
    df_test = pd.concat([stars_test,qsos_test], sort=False)

    if save:
        df_train.to_hdf(str(save_prefix)+'train.hdf5','data')
        df_test.to_hdf(str(save_prefix)+'test.hdf5','data')

    if concat:
        return df_train, df_test
    else :
        return stars_train, qsos_train, df_test

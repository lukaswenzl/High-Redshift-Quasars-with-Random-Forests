import numpy as np
import pandas as pd

# Load the random forest classification functions
import random_forest as rf
# Load the Machine Learning
#from sklearn.ensemble import RandomForestClassifier


#for parsing the arguments for the file
import argparse

def create_classifiers(testmode=0, cores=4):
    """Create the random forest classifiers used in the pipeline.

    All details are hardcoded.

    Parameters
    ----------
    testmode: if set to one the algorithm only uses a subsample for
        training for quick testing
    server: set to the number of cores if running on a powerfull machine with
        large RAM, otherwise 4 cores are used.

    """
    if (cores == 0):
        cores = 4 #how many cores to use for the training


    print("loading catalogs")

    df_all_quasars= pd.read_csv("data/training/quasars_dr14_and_highz.csv")
    df_all_stars=pd.read_csv("data/training/stars_sdss_and_dwarfs.csv")

    #to test ###########################################
    if (testmode==1):
        df_all_stars = df_all_stars.sample(n=10000)
        df_all_quasars = df_all_quasars.sample(n=10000)
        #df_sim = df_sim.sample(n=10000)

    df_train = pd.concat([df_all_stars,df_all_quasars], sort=False)

    print("Excluding two stripes")
    start_shape = df_train.shape[0]
    df_excluded = df_train.query("(ps_ra <= 60 or ps_ra >= 300) and (ps_dec <= 1.26 and ps_dec >= -1.26) ") #used for observation
    #df_excluded = df_train.query("(ps_ra <= 60 or ps_ra >= 300) and (ps_dec <= 5 and ps_dec >= -5) ")

    df_train = df_train.query("ps_ra > 60 and ps_ra < 300 and (ps_dec> 1.26 or ps_dec < -1.26) ").copy()
    #df_train = df_train.query("(ps_ra > 60 and ps_ra < 300) or (ps_dec> 5 or ps_dec < -5) ").copy()

    print("{} elements where excluded. Here are the value counts counts:".format(start_shape-df_train.shape[0]))
    print(df_excluded['mult_class_true'].value_counts())
    print("---")

    print("Here are the value counts for the training data:")
    print(df_train['mult_class_true'].value_counts())

    features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']
    label = 'mult_class_true'


    #params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 25,
    #        'min_samples_split': 2, 'n_estimators': 200,'random_state': 1,'n_jobs': cores}#n_jobs: how many processors, random_state: give me the same random variables for reproducability

    #optimized params for stackmag 5th october
    params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 25,
            'min_samples_split': 3, 'n_estimators': 300,'random_state': 1,'n_jobs': cores}#n_jobs: how many processors, random_state: give me the same random variables for reproducability


    rand_state = 1

    rf.rf_class_create(df_train, features, label, params, rand_state, save=True, save_filename="data/rf_classifier_PS_ALLWISE")

    # print("-------------------------------------------------------------")
    # print("finished, now again with non quasar list")
    # print("-------------------------------------------------------------")
    #
    #
    # df_train = pd.concat([df_all_stars,df_all_quasars,df_not_quasars], sort=False)
    #
    #
    # features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']  ##shoudl i use PS_z?
    # label = 'mult_class_true'
    #
    #
    # #params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 25,
    # #        'min_samples_split': 2, 'n_estimators': 200,'random_state': 1,'n_jobs': cores}#n_jobs: how many processors, random_state: give me the same random variables for reproducability
    #
    # #optimized params for stackmag 5th october
    # params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 25,
    #         'min_samples_split': 3, 'n_estimators': 300,'random_state': 1,'n_jobs': cores}
    #
    # rand_state = 1
    #
    # rf_class.rf_class_create(df_train, features, label, params, rand_state, save=True, save_filename="data/rf_classifier_with_Not_quasars_PS_ALLWISE")
    #

    # print("-------------------------------------------------------------")
    # print("finished, now again with simulated quasars")
    # print("-------------------------------------------------------------")
    #
    # df_train = pd.concat([df_all_stars,df_sim, df_sim_low], sort=False) #removed not quasars from sim class
    #
    # #optimized parameters old
    # #params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 30,
    # #        'min_samples_split': 3, 'n_estimators': 200,'random_state': 1,'n_jobs': cores}#n_jobs: how many processors, random_state: give me the same random variables for reproducability
    #
    # #optimized parameters old
    # params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 25,
    #         'min_samples_split': 3, 'n_estimators': 300,'random_state': 1,'n_jobs': cores}#n_jobs: how many processors, random_state: give me the same random variables for reproducability
    #
    #
    #
    # rand_state = 1
    # label = 'mult_class_true'
    #
    #
    #
    # rf_class.rf_class_create(df_train, features, label, params, rand_state, save=True, save_filename="data/rf_classifier_sim_QSO_PS_ALLWISE")
    #

def create_regressors(testmode=0, cores=4):
    """Create the random forest regressors used in the pipeline.

    All details are hardcoded.

    Parameters
    ----------
    testmode: if set to one the algorithm only uses a subsample for
        training for quick testing
    server: set to the number of cores if running on a powerfull machine with
        large RAM, otherwise 4 cores are used.

    """
    if (cores == 0):
        cores = 4 #how many cores to use for the training


    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    print("loading catalogs")

    df_train=  pd.read_csv("data/training/quasars_dr14_and_highz.csv")

    # Possibility of running on a subsample for testing
    if (testmode==1):
        df_train = df_train.sample(frac=0.1)

    print("Excluding two stripes")
    start_shape = df_train.shape[0]
    df_train = df_train.query("ps_ra > 60 and ps_ra < 300 and (ps_dec> 1.26 or ps_dec < -1.26) ").copy() #used for observation
    #df_train = df_train.query("(ps_ra > 60 and ps_ra < 300) or (ps_dec> 5 or ps_dec < -5) ").copy()
    print("removed {} objects".format(start_shape-df_train.shape[0]))

    # --------------------------------------------------------------------------
    # Random Forest Regression
    # --------------------------------------------------------------------------
    print("creating regressor for full range")

    features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']

    label = 'z'
    rand_state = 1

    #optimized parameters
    #params = {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 2, 'n_jobs': cores,
    #          'random_state':rand_state,}

    #optimized parameters stackmag 5th october 2018
    params = {'n_estimators': 400, 'max_depth': 20, 'min_samples_split': 3, 'n_jobs': cores,
              'random_state':rand_state,}


    rf.rf_create_reg(df_train,features,label,params,rand_state,save=True,save_filename='data/rf_regressor_full_PS_ALLWISE')

    #---------------------------------------------------------------------------
    #highz regression
    #---------------------------------------------------------------------------
    print("highz regression")
    print("dropping all objects with z<=4.5")
    df_train_highz = df_train[df_train.z >4.5].copy(deep=True)

    #features, label and rand_state are the same

    #old optimized
    #params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2, 'n_jobs': cores,
    #          'random_state':rand_state,}

    #optimized parameters 5th october 2018
    params = {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 3, 'n_jobs': cores,
              'random_state':rand_state,}

    df_train_highz.info()
    rf.rf_create_reg(df_train_highz,features,label,params,rand_state,save=True,save_filename='data/rf_regressor_highz_PS_ALLWISE')

# def create_sim_regressors(testmode=0, cores =4):
#     """Create the random forest regressors used in the pipeline that use the simulated data.
#
#     All details are hardcoded.
#
#     Parameters
#     ----------
#     testmode: if set to one the algorithm only uses a subsample for
#         training for quick testing
#     server: set to the number of cores if running on a powerfull machine with
#         large RAM, otherwise 4 cores are used.
#
#     """
#     if (cores== 0):
#         cores = 4 #how many cores to use for the training
#
#     # --------------------------------------------------------------------------
#     # Preparing the feature matrix
#     # --------------------------------------------------------------------------
#     print("loading simulated catalogs")
#
#     df_sim = pd.read_csv("....where is it?.csv")
#
#     # Possibility of running on a subsample for testing
#     if (testmode ==1):
#         df_sim = df_sim.sample(frac=0.05)
#
#     # --------------------------------------------------------------------------
#     # Random Forest Regression
#     # --------------------------------------------------------------------------
#
#     features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']
#
#     label = 'z' # Z_VI
#     rand_state = 1
#
#     print("#-------------------------------------------------------------------")
#     print("#fitting regressor on full simulated datset")
#     print("#-------------------------------------------------------------------")
#
#     #optimized params
#     params = {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 2, 'n_jobs': cores,
#               'random_state':rand_state,}
#
#     df_sim[features].info()
#     rf.rf_create_reg(df_sim,features,label,params,rand_state,save=True,save_filename='data/rf_regressor_sim_full_PS_ALLWISE')
#
#
#     print("#-------------------------------------------------------------------")
#     print("#fitting regressor on highz simulated datset")
#     print("#-------------------------------------------------------------------")
#
#     df_sim = df_sim.query("z>4.5")
#     features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']
#
#     #optimized param
#     params = {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 3, 'n_jobs': cores,
#               'random_state':rand_state,}
#
#     df_sim[features].info()
#     rf.rf_create_reg(df_sim,features,label,params,rand_state,save=True,save_filename='data/rf_regressor_sim_PS_ALLWISE')

def parseArguments():
    """Parse the given Arguments when calling the file from the command line.

    Returns
    -------
    arg
        The result from parsing.

    """
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-t", "--test", help="Optionally test algorithm on subsample of the data. Set to 1 for testing", type=int, default=0)

    parser.add_argument("--cores", help="Optimized code for a server with a lot of RAM, set to the number of available cores", type=int, default=40)


    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 2.0') #version 1.0 is for the observations in June 2018
    #version 1.1 contains the optimizations made after the june observations (mainly the switch to stackmags)
    #version 1.2 changed sim class to NOT include the list of failed candidates (not qsos)
    #... copied changes made to crossval version
    #version 1.5 added check for duplicate quasars and remove them
    #version 1.6 new simulated quasars (december)
    ##-------------------
    #version 2.0: combined training of classifier and regressor, streamlined input
    #version 2.1: Tryied to updates excluded area to a little more than stripe 82 but decided not to keep it, so no change

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parseArguments()
    print("Version 2.1: ml_quasar")

    create_classifiers(args.test, args.cores);

    create_regressors(args.test, args.cores);

    print("currently no simulated regressors and classifiers planed")
    #additional regressors from simulated data
    #create_sim_regressors(args.test, args.cores);
    print("------------------ finished ------------------")

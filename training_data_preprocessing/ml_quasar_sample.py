##Edit Lukas:added L, K Stars to classification
##added prepare_color_catalog
import pandas as pd
import numpy as np
import math

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def prepare_flux_ratio_catalog(cat,passband_names,sigma=False):
    """ Calculating the flux ratios from the fluxes provided by
        the input df and dropping all rows with NaN values in the
        process to ensure a full data set

    Input:
            cat (DataFrame) as the input flux catalog
            passband_names (list) of the filter names considered
                for calculating the flux ratios
    Output:
            df (DataFrame) catalog including the flux ratios
            flux_ratio_names (list) list of the labels for
                the flux ratio columns
    """

    df = cat.copy(deep=True)

    # Drop all rows with Inf and NaN values in the passband considered
    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.dropna(axis=0,how='any',subset=passband_names,inplace=True)

    # Calculate the flux ratios and add them to the dataframe
    flux_ratio_names = []
    flux_ratio_err_names= []



    if sigma :

        for name in passband_names:
            df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)

        for i in range(len(passband_names)-1):

            passband_a = np.array(df[passband_names[i]])
            passband_b = np.array(df[passband_names[i+1]])
            sigma_a = np.array(df['sigma_'+passband_names[i]])
            sigma_b = np.array(df['sigma_'+passband_names[i+1]])

            passband_a_name = passband_names[i].split('_')[1]
            passband_b_name = passband_names[i+1].split('_')[1]

            df[str(passband_a_name+passband_b_name)] = \
            passband_a / passband_b

            flux_ratio_names.append(str(passband_a_name+passband_b_name))

            df[str('sigma_'+passband_a_name+passband_b_name)] = \
            np.sqrt((sigma_a/passband_b)**2 + (passband_a/passband_b**2*sigma_b))

            flux_ratio_err_names.append('sigma_'+ \
            str(passband_a_name+passband_b_name))

    else :

        #for name in passband_names:
            #df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)

        for i in range(len(passband_names)-1):

            passband_a = np.array(df[passband_names[i]])
            passband_b = np.array(df[passband_names[i+1]])
            # sigma_a = np.array(df['sigma_'+passband_names[i]])
            # sigma_b = np.array(df['sigma_'+passband_names[i+1]])

            passband_a_name = passband_names[i].split('_')[1]
            passband_b_name = passband_names[i+1].split('_')[1]

            df[str(passband_a_name+passband_b_name)] = \
            passband_a / passband_b

            flux_ratio_names.append(str(passband_a_name+passband_b_name))

            # df[str('sigma_'+passband_a_name+passband_b_name)] = \
            # np.sqrt((sigma_a/passband_b)**2 + (passband_a/passband_b**2*sigma_b))

            # flux_ratio_err_names.append('sigma_'+ \
            # str(passband_a_name+passband_b_name))




    return df, flux_ratio_names

def prepare_color_catalog(cat,passband_names,sigma=False):
    """ Calculating the colors from the mags provided by
        the input df and dropping all rows with NaN values in the
        process to ensure a full data set. The bands are expected to be named
        <something>_mag_<band>

    Input:
            cat (DataFrame) as the input flux catalog
            passband_names (list) of the filter names considered
                for calculating the flux ratios
    Output:
            df (DataFrame) catalog including the flux ratios
            flux_ratio_names (list) list of the labels for
                the flux ratio columns
    """

    df = cat.copy(deep=True)

    # Drop all rows with Inf and NaN values in the passband considered
    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.dropna(axis=0,how='any',subset=passband_names,inplace=True)

    # Calculate the flux ratios and add them to the dataframe
    flux_ratio_names = []
    flux_ratio_err_names= []



    if sigma :

        for name in passband_names:
            df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)

        for i in range(len(passband_names)-1):

            passband_a = np.array(df[passband_names[i]])
            passband_b = np.array(df[passband_names[i+1]])
            sigma_a = np.array(df['sigma_'+passband_names[i]])
            sigma_b = np.array(df['sigma_'+passband_names[i+1]])

            passband_a_name = passband_names[i].split('_mag_')[1]
            passband_b_name = passband_names[i+1].split('_mag_')[1]

            df[str(passband_a_name+'-'+passband_b_name)] = \
            passband_a - passband_b

            flux_ratio_names.append(str(passband_a_name+'-'+passband_b_name))

            df[str('sigma_'+passband_a_name+'-'+passband_b_name)] = \
            np.sqrt((sigma_a)**2 + (sigma_b)**2) ##I think thats how to calc the error
            #I don't use this so I havent tested it

            flux_ratio_err_names.append('sigma_'+ \
            str(passband_a_name+'-'+passband_b_name))

    else :

        #for name in passband_names:
            #df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)

        for i in range(len(passband_names)-1):

            passband_a = np.array(df[passband_names[i]])
            passband_b = np.array(df[passband_names[i+1]])
            # sigma_a = np.array(df['sigma_'+passband_names[i]])
            # sigma_b = np.array(df['sigma_'+passband_names[i+1]])

            passband_a_name = passband_names[i].split('_mag_')[1]
            passband_b_name = passband_names[i+1].split('_mag_')[1]

            df[str(passband_a_name+'-'+passband_b_name)] = \
            passband_a - passband_b

            flux_ratio_names.append(str(passband_a_name+'-'+passband_b_name))

            # df[str('sigma_'+passband_a_name+passband_b_name)] = \
            # np.sqrt((sigma_a/passband_b)**2 + (passband_a/passband_b**2*sigma_b))

            # flux_ratio_err_names.append('sigma_'+ \
            # str(passband_a_name+passband_b_name))




    return df, flux_ratio_names


def build_full_sample(df_stars, df_quasars, star_qso_ratio):

    """ Merging the star and quasar flux_ratio catalogs according to
    the set variable star_quasar_ratio. This is the first step to create
    more realistic data set, since the intrinsic ratio of stars to quasars
    will not be mimicked by simply combining both data sets. The catalogs
    are labelled dependend on their origin catalog.

    TO DO:
    This function should be expanded in order to return a DataFrame that
    mimicks the intrinsic quasar/star distribution as good as possible.

    Parameters:
            df_stars : pandas dataframe
            Star flux ratio catalog

            df_quasars : pandas dataframe
            Quasar flux ratio catalog

            star_qso_ratio : integer
            Goal ratio of stars to quasars

    Returns:
            df : pandas dataframe
            Merged flux ratio catalog with specified star to quasar ratio
    """

    df_quasars['label'] = 'QSO'
    df_stars['label'] = 'STAR'

    if df_stars.shape[0] > df_quasars.shape[0]*star_qso_ratio:
        # calculate number of objects to sample
        sample_size = df_quasars.shape[0]
        star_sample = df_stars.sample(sample_size*star_qso_ratio)
        qso_sample = df_quasars.sample(sample_size)

        df = pd.concat([qso_sample,star_sample], sort=False)
    else:
        # calculate number of objects to sample
        sample_size = df_stars.shape[0]

        star_sample = df_stars.sample(int (sample_size))
        qso_sample = df_quasars.sample(5)#int (round(sample_size/star_qso_ratio)))
        df = pd.concat([qso_sample,star_sample], sort=False)

    return df



def create_star_labels(df_stars, label_name, star_label):
    """ This function creates a new column for the stellar classes and either
    manually deletes objects within certain classes to retain only classes with
    large numbers of objects or deletes such classes that have less than 400
    objects. The option is hardcoded in the routine.

    Parameters:
            df_stars : pandas dataframe
            Star flux ratio catalog

            label_name : string
            Name of the new general classification column

            star_label : string
            Name of the old stellar classification column

    Returns:
            df_stars : pandas dataframe
            Updated dataframe with a new star class column and only objects in
            specified classes.
    """

    # Create a new column for the stellar classifications and copy them
    df_stars[label_name] = df_stars[star_label]
    # Create a list of the names of the star classes
    star_classes = df_stars[label_name].value_counts().index

    # Exclude classes with less than 400 objects in class
    # for label in star_labels:

        # if df_stars.class_label.value_counts()[label] < 400:
        #     df_stars.drop(df_stars.query('class_label == "'+label+'"').index,
        #                                                         inplace=True)

    # Manually exclude a number of star classes to leave only the ones with
    # a large number of objects in
    #labels_to_exclude = ['O','B','OB','L','T','WD','CV','Carbon']
    labels_to_exclude = ['O','B','Y','OB','WD','CV','Carbon'] #for higher z, added Y

    for label in labels_to_exclude:
        df_stars.drop(df_stars.query(str(label_name)+' =="'+str(label)+'"').index,
                                                            inplace=True)

    # Delete objects with class == "null"
    df_stars.drop(df_stars.query(str(label_name)+' == "null"').index,
                                                        inplace=True)
    df_stars.dropna(subset=[label_name],inplace=True)

    # Rename labels
    # for star_class in star_classes:
    #     idx = df_stars.query(str(label_name)+' =="'+str(star_class)+'"').index
    #     df_stars.loc[idx,label_name] = str(star_class)+'_star'

    # Print the value counts for the new star class column
    #print("Stellar classes with new labels: \n ")
    print(df_stars[label_name].value_counts())

    return df_stars


def create_qso_labels(df_qsos, label_name, z_label):
    """ This function creates a new column for quasar classes and uses the
    redshift column specified by z_label to sort the objects into the four
    redshift classes specified in the routine.

    Parameters:
            df_qsos : pandas dataframe
            Quasar flux ratio catalog

            label_name : string
            Name of the new general classification column

            z_label : string
            Name of the redshift column

    Returns:
            df_qsos : pandas dataframe
            Updated dataframe with a new quasar classification column.
    """

    # lower and upper redshift boundaries
    #lowz=[0,1.5,2.2,3.5]
    #highz=[1.5,2.2,3.5,10]

    #lowz=[0,4.7,5.1,5.7] Lukas first try, not enough high redshift objects
    #highz=[4.7,5.1,5.7,10]

    lowz=[0,1.5,3.5,4.7]
    highz=[1.5,3.5,4.7,10]

    # names of the redshift class labels
    # labels=['0<z<=1.5','1.5<z<=2.2','2.2<=3.5','3.5<z']
    labels=['vlowz','lowz','midz','highz']

    # create new column and fill it with the value "null"
    df_qsos[label_name] = 'null'

    # Reduce dataframe only to objects that have redshifts in a sane range
    df_qsos.query('0<'+str(z_label)+'<10',inplace=True)

    # Set the classes using the redshift bins specified above
    for idx in range(len(lowz)):

        df_qsos.loc[
                df_qsos.query(str(lowz[idx])+'<'+z_label+'<='+str(highz[idx])).index, \
            label_name] = labels[idx]

    # Delete objects with class == "null"
    df_qsos.drop(df_qsos.query(str(label_name)+' == "null"').index,
                                                        inplace=True)

    # Print the value counts for the new star class column
    #print("Quasar classes with new labels: \n")
    print(df_qsos[label_name].value_counts())

    return df_qsos


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

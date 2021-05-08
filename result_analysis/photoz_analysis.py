import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LogNorm
#from sklearn.learning_curve import validation_curve

def evaluate_photoz(z_true, z_phot):
    """
    This function evaluates the accuracy of the predicted photometric redshifts.
    It calculates the fraction of objects within a 0.1/0.2/0.3 redshift distance
    to the true values, the fraction of objects within a relative redshift
    distance of 0.1/0.2/0.3 (dz01, dz02, dz03) and the standard deviation and
    the relative redshift standard deviation. All but the dz0* values are
    printed on the screen.

    Parameters:
        z_true : array-like, shape (n_samples)
        Array containing the true values of the regression

        z_phot : array-like, shape (n_samples)
        Array containing the predicted values of the photometric redshift
        estimation.

    Returns:
        dz01 : float
        Fraction of predicted values within 0.1 in relative redshift distance to
        the true values.

        dz02 : float
        Fraction of predicted values within 0.2 in relative redshift distance to
        the true values.

        dz03 : float
        Fraction of predicted values within 0.3 in relative redshift distance to
        the true values.
    """

    delta_z = abs(z_phot - z_true)

    print("Standard deviation: ")
    print(np.std(z_phot - z_true))

    dz03 = float(len(np.where(delta_z < 0.3)[0]))/float(len(delta_z))
    dz02 = float(len(np.where(delta_z < 0.2)[0]))/float(len(delta_z))
    dz01 = float(len(np.where(delta_z < 0.1)[0]))/float(len(delta_z))

    print("R 0.3 : ",dz03)
    print("R 0.2 : ",dz02)
    print("R 0.1 : ",dz01)

    delta_z = abs(z_phot - z_true)/(1 + z_true)

    print("Redshift normalized standard devation: ")
    print(np.std((z_phot - z_true)/(1 + z_true)))

    dz03 = float(len(np.where(delta_z < 0.3)[0]))/float(len(delta_z))
    dz02 = float(len(np.where(delta_z < 0.2)[0]))/float(len(delta_z))
    dz01 = float(len(np.where(delta_z < 0.1)[0]))/float(len(delta_z))

    print("Photometric Redshift evaluation")
    print("Total number of test objects : ", len(delta_z))
    print("Delta z < 0.3 : ",dz03)
    print("Delta z < 0.2 : ",dz02)
    print("Delta z < 0.1 : ",dz01)

    return dz03,dz02,dz01


def plot_redshifts(y_true,y_pred,title='Photometric Redshifts'):
    """
    This function creates a redshift-redshift plot of measured redshifts against
    photometrically determined redshifts. It returns the matplotlib plot.

    Parameters:
        y_test : array-like, shape (n_samples)
        Array containing the true values of the regression

        y_pred : array-like, shape (n_samples)
        Array containing the predicted values of the regression

    Returns:
        plt : matplotlib pyplot element

    """

    # Tex font
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    fig = plt.figure(num=None,figsize=(6,6), dpi=140)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.96)
    ax = fig.add_subplot(1,1,1)

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    cmap = plt.cm.Blues
    cmap.set_bad('0.85',1.0)

    #replaced [[0,6], [0,6]] Lukas
    cax = plt.hist2d(y_true,y_pred, bins=240,range =  [[0,8], [0,8]],
    norm = LogNorm(), cmap=cmap, zorder=0)
    cbar = plt.colorbar(ticks=[1,5,10,20,40,80])
    cbar.ax.set_yticklabels([r'$1$',r'$5$',r'$10$',r'$20$',r'$40$',r'$80$'])
    cbar.set_label(label=r'$\rm{Number\ of\ objects}$',size=20)


    ax.plot(np.arange(8),np.arange(8),'k',linewidth=1)


    ax.set_xlabel(r'$\rm{Measured\ redshift}$',fontsize =20)
    ax.set_ylabel(r'$\rm{Photometric\ redshift\ estimate}$', fontsize = 20)

    ax.set_xlim(0,7)
    ax.set_ylim(0,7)

    ax.set_title(title)

    return plt

def plot_error_hist(y_true,y_pred):
    """
    This function creates a histogram of the photometrically predicted redshift
    values against the true values of the data set. It calls the evaluate_photoz
    function to get the fraction of objects within a relative redshift distance
    of 0.1/0.2/0.3 (dz01, dz02, dz03).

    Parameters:
        y_test : array-like, shape (n_samples)
        Array containing the true values of the regression

        y_pred : array-like, shape (n_samples)
        Array containing the predicted values of the regression

    Returns:
        plt : matplotlib pyplot element

    """

    # Tex font
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    diff = np.array(y_true-y_pred)

    dz03, dz02, dz01 = evaluate_photoz(y_true, y_pred)

    fig = plt.figure(num=None,figsize=(6,6), dpi=140)
    fig.subplots_adjust(left=0.14, bottom=0.1, right=0.95, top=0.95)
    ax = fig.add_subplot(1,1,1)


    ax.hist(diff,bins=500, histtype='stepfilled',facecolor='blue',alpha=0.6)
    ax.text(0.05,0.9,r'$\Delta z < 0.3\ :\ $'+r'${0:.2f}\%$'.format(dz03*100)
            ,fontsize=15,transform = ax.transAxes)
    ax.text(0.05,0.83,r'$\Delta z < 0.2\ :\ $'+r'${0:.2f}\%$'.format(dz02*100)
          ,fontsize=15,transform = ax.transAxes)
    ax.text(0.05,0.76,r'$\Delta z < 0.1\ :\ $'+r'${0:.2f}\%$'.format(dz01*100)
          ,fontsize=15,transform = ax.transAxes)

    ax.set_xlim(-0.6,0.6)

    ax.set_xlabel(r'$\Delta z$',fontsize =20)
    ax.set_ylabel(r'$\rm{Number\ of\ objects\ per\ bin}$', fontsize = 20)

    return plt

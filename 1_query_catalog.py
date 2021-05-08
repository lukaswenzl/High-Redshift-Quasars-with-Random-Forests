#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os
#import errno
import time

import numpy as np
from astropy.io import ascii, fits
#from astropy.table import Table
from astropy.utils.console import human_time

from lsdlib import colunits, col_stack_all3, create_table, drop_table
from lsdlib import lsd_query, lsd_create_header, lsd_import, read_input
#from lsdlib import get_coldef_from_dtype
#from lsddb_from_file import create_file_db
#from astrotools import nmgy2abmag, zscale

#from util_efs import query_lsd
#import lsd.bounds as b
from lsd import DB
from lsd import bounds as lsdbounds

import warnings

EXAMPLES = '''
python 1_query_catalog.py -o data/catalog/stripe82.fits -a "stripe82"
python 1_query_catalog.py -a "test" -o data/catalog/test.fits
python 1_query_catalog.py -a "full" -o data/catalog/full.fits > data/catalog/full.log &

'''
#EXAMPLES ='''
#OUTDATED
# To get the SDSS information of the objects in one table:
#     get_info_from_cat.py -f zdrops_wise_first.fits --dbname zdrops_first_wise_sdss --dbpath ./ -o zdrops_first_wise_sdss.fits -c sdss -e 1
#
#         get_info_from_cat.py -f $DROPBOX/ps1_results/known_hqz/ebt/known_hzq.cat --dbname known_qso_nov13 --dbpath ./ -o hzq_allinfo_nov13.fits -c all --where "primary_avg==1"
#
# -Cross-match existing fits table with 2MASS and ALLWISE
#     get_info_from_cat.py -f idrops_hz_stack_all3_noM31_clean.fits -c wise,twomass --dbpath ./ --dbname idrops_hz_stack_all3_wise_2mass -o idrops_hz_stack_all3_dbs.fits --no-all
#
# -Cross-match existing fits table with ALLWISE
#     get_info_from_cat.py -f zdrops_stack_all3_noM31.fits -c wise --dbpath ./ --dbname zdrops_stack_all3_allwise -o zdrops_stack_all3_allwise.fits --no-all
#
# -Cross match existing file with ALLWISE and PS1 and getting only important columns. With a where statement.
#     get_info_from_cat.py -f test -c ps1_pv1,wise --dbpath ./ --dbname testing -o testing3.fits --no-all --no-fast --where "primary_avg==1"
#
# -Get WISE and PS1 info for all the known QSOs. File from Emanuele. In order to get all the qsos
# a radius of 4 arcsecs for PS1 and 5 arcsecs for WISE is used.
#     get_info_from_cat.py -f  ~/Dropbox/qso_stars_colors/spectroscopic_qsos/QSOlist_degree.dat -c ps1_pv1,wise --dbpath ./ --dbname qsos_all -o qsos_ps1_wise.fits --no-all --no-fast --where "primary_avg==1" --ps1_radius 4.0 --wise_radius 5.0
#
#
#   Get Quasar info creating a temporal database and getting only important fields from PS1, WISE, FIRST, 2MASS and UKIDSS
#
#   get_info_from_cat.py -f  $DROPBOX/ps1_results/known_hqz/known_hzq.cat --dbpath ./ -o hzq_allinfo_dec13.fits -c wise,twomass,first,ps1_pv1,ukidss --where "primary_avg==1" --dbname known_qso_dec13 --no-all --no-fast
#
#
#  get_info_from_cat.py -f  $DROPBOX/ps1_results/known_hqz/known_hzq.cat --dbpath ./ -o hzq_allinfo_may14.fits -c all --where "primary_avg==1" --dbname known_qso_may14 --no-all --no-fast
#
#  get_info_from_cat.py -f HVA_201409.dat --dbpath ./ -o HVA_201409_PV2.fits -c ps1_pv2 --where "primary_avg==1" --dbname HVAsep2014 --no-all --fast
#
#  Get EBV for known QSOs:
#  get_info_from_cat.py -f known_hzq.csv --dbpath ./ -o known_hzq_nov2015_ebv.fits  --dbname knownqsos -c 'ebv' --no-fast
#
#  Get EBV and PS1 PV3 info for known QSOs
#  get_info_from_cat.py -f known_hzq_nov2015coord.csv --dbpath ./ -o known_hzq_nov2015_ebv_pv3.fits  --dbname knownqsos -c 'ebv,ps1_pv3' --no-fast --no-all
#
# TODO
#     The option --fast is not working. Use in the meantime only --no-fast  !
#     SOLVED?? Test it. Now that I update it to be compatibe with PV2 there are problems again.
#
#          '''

#based on code from E. Schlafly
def query_lsd(querystr, db=None, bounds=None, **kw):
    if db is None:
        db = os.environ['LSD_DB']
    if not isinstance(db, DB):
        dbob = DB(db)
    else:
        dbob = db
    if bounds is not None:
        bounds = lsdbounds.make_canonical(bounds)
    query = dbob.query(querystr, **kw)
    return query.fetch(bounds=bounds)


def syntax_for_single_columns(basename, catalogname=None, part2="" , filters=["g", "r", "i", "z", "y"]):
    """Convert a multicolumn to single column
    part2 is only used if formulas are applied to column"""
    string= ""
    if(catalogname==None):
        catalogname = basename
    for i in range(len(filters)):
        if(filters[i] != None):
            string += ", "
            constructed_name = catalogname + ".T["+str(i)+"]"+part2+" "
            constructed_name += "as "+filters[i]+basename
            string += constructed_name
    return string

def signal_to_noise_single_columns():
    filters=["g", "r", "i", "z", "y"]
    string= ""
    basename="_sn"
    for i in range(len(filters)):
        if(filters[i] != None):
            string += ", "
            constructed_name = "1. /np.abs(flux_psf_stk_err.T["+str(i)+ "]/flux_psf_stk.T["+str(i)+"]) "
            constructed_name += "as "+filters[i]+basename
            string += constructed_name
    return string


#IMPORTANT FIELDS FOR DATABASES
def get_ps1_important(ps1):
    #ps1_important = "obj_id, "
    ps1_important = '{0:s}.ra_stk as ra_ps1_stack,{0:s}.dec_stk as dec_ps1_stack'.format(ps1)
    ps1_important+= ',{0:s}.ra as ps_ra,{0:s}.dec as ps_dec, '.format(ps1)
    ps1_important+='psf_qf,psf_qf_perf, stargal_sep, number_pos, nmeasure,'
    ps1_important+='projection_id,  nwarp_ok, flags, cat_id'#skycell_id, removed because it has a weird formatter_class
    ps1_important+= syntax_for_single_columns("PSFStackMag", "mag_psf_stk")#first is the base for the column names I want in the final fits file, second argument is name in catalog
    ps1_important+=signal_to_noise_single_columns()
    ps1_important+=syntax_for_single_columns("_mag_lim", "-2.5*np.log10(3 * np.abs(flux_psf_stk_err", part2="))+8.9")
    ps1_important+= syntax_for_single_columns("MeanPSFMag", "mag")
    #ps1_important+='mag_psf_stk, np.abs(flux_psf_stk_err/flux_psf_stk) as f_sig,'
    #ps1_important+='1./f_sig as sn, '
    #ps1_important+='-2.5*np.log10(3 * np.abs(flux_psf_stk_err))+8.9 as mag_lim,'
    ps1_important+= ", mag_psf_stk.T[3]-mag_kron_stk.T[3] as psf_kron, mag_psf_stk.T[3]-mag_ap_stk.T[3] as psf_ap" #only using extension in z band
    #ps1_important+=',mag_ap_stk, mag_kron_stk, mag_psf_stk-mag_kron_stk as psf_ap,'#where (stack_primary_off==1) & (psf_ap < 0.3) & (psf_ap > -0.3) & ((glat > 20) | (glat< 20))
    #ps1_important+='mag_ap_stk - mag_psf_stk as f_ext,'
    ps1_important+= syntax_for_single_columns("nstack")
    ps1_important+= syntax_for_single_columns("nstack_det")
    ps1_important+= syntax_for_single_columns("stack_primary_off")
    ps1_important+= syntax_for_single_columns("stack_best_off")
    ps1_important+= syntax_for_single_columns("ubercal_dist")
    ps1_important+= syntax_for_single_columns("psf_qf_perf_max")
    #ps1_important+='nstack, nstack_det, stack_primary_off,'
    #ps1_important+='stack_best_off, ubercal_dist'
    ps1_important+= ', photflags_u, photflags_l'

    return ps1_important

wise_important=', allwise.designation as wise_designation, allwise.coadd_id as coadd_id, allwise.ra as ra_wise, allwise.dec as dec_wise'
wise_important+=', allwise.w1mpro as w1mpro, allwise.w1sigmpro as w1sigmpro'
wise_important+=', allwise.w1snr as w1snr, allwise.w2mpro as w2mpro, allwise.w2sigmpro as w2sigmpro'
wise_important+=', allwise.w2snr as w2snr, allwise.w3mpro as w3mpro, allwise.w3sigmpro as w3sigmpro'
wise_important+=', allwise.w3snr as w3snr, allwise.w4mpro as w4mpro, allwise.w4sigmpro as w4sigmpro'
wise_important+=', allwise.w4snr as w4snr'#, allwise.w1mpro+2.699 as w1ab, allwise.w2mpro+3.339 as w2ab'
#wise_important+=', allwise.w3mpro+5.174 as w3ab, allwise.w4mpro+6.620 as w4ab'
wise_important+= ', nb, na' #nb number of psf for fit, should be 1; na: 0 for not blended, 1: for activly blended
                            #see:http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_2a.html
#wise_important+= syntax_for_single_columns("cc_flag", "cc_flags", filters=["w1", "w2", "w3", "w4"])
#', allwise.cc_flags as cc_flags' #do i need this?

# wiserej_important=',allwiserej.ra as ra_wiserej, allwiserej.dec as dec_wiserej'
# wiserej_important+=',allwiserej.w1sigmpro as w1sigmpro_wiserej,  allwiserej.w1snr as w1snr_wiserej, allwiserej.w2sigmpro as w2sigmpro_wiserej, allwiserej.w2snr as w2snr_wiserej, '
# wiserej_important+='allwiserej.w3sigmpro as w3sigmpro_wiserej, allwiserej.w3snr as w3snr_wiserej, allwiserej.w4sigmpro as w4sigmpro_wiserej, allwiserej.w4snr as w4snr_wiserej'
# wiserej_important+=', allwiserej.w1mpro+2.699 as w1ab_wiserej, allwiserej.w2mpro+3.339 as w2ab_wiserej, allwiserej.w3mpro+5.174 as w3ab_wiserej, allwiserej.w4mpro+6.620 as w4ab_wiserej'
# wiserej_important+=', allwiserej.cc_flags as cc_flags_wiserej, allwiserej.nb as nb_wiserej'

# #EBV
# select_ebv=',equgal(ra,dec) as (l,b), SFD.EBV(l,b) as ebv, '
# select_ebv+='ebv*3.172 as ebv_g, ebv*2.271 as ebv_r, ebv*1.682 as ebv_i,'
# select_ebv+='ebv*1.322 as ebv_z, ebv*1.087 as ebv_y'




def parse_arguments():

    parser = argparse.ArgumentParser(
        description='''
        access PV34 (default) and crossmatch with
        WISE: AllWISE vNov13,2013
        2mass
        first
        sdss
        ukidss DR10 source catalog
        galex (Galex AIS Two-band UV all sky catalogue from the Galaxy Evolution Explorer)



        The limiting magnitudes in the PS1 catalog are 3 sigma lim mag' (where is that?)

        So you will end up with a text or fits file containing the
        ps1/WISE and/or 2MASS information of the objects you are interested in.

                      ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)


    parser.add_argument('-c', '--catalog', default='ps1_pv34', type=str,
        #choices=['ps1_pv1', 'wise', 'first', '2mass', 'sdss',
        #        'wise,2mass', 'all'],
        help="Catalog from which the information is retrieved\
        (default: %(default)s, choices=['ps1_pv1', 'ps1_pv2', 'ps1_pv3', 'ps1_pv34',\
        'allwise','wisereject', 'unwisew12', 'unwise_prelim_w1', 'unwise_prelim_w2', 'first', 'twomass',\
         'sdss', 'ukidss', 'galex', 'lap_pv1', 'uhs', 'nvss', 'ebv','vhs_j', 'vhs_y', 'vhs_h', 'vhs_ks', 'vhs_dr4p1',\
          'decals_dr2', 'decals_dr3', 'decals_dr4', 'decals_dr5', 'decals_dr6', 'decals_dr7', 'des_dr1', 'all']. If all is selected, the default PS1 catalog is queried" )


    # parser.add_argument('-n', '--dbname', type=str, required=True,
    #     help='Name of the new table (LSD database) with the\
    #     interesting objects to be created')

    # parser.add_argument('-p', '--dbpath', type=str, required=True,
    #     help='path where the new table with the interesting objects and\
    #     the files with the extra ps1 information are going to be created')
    #i instead use export LSD_DB=/data/beegfs/astro-storage/groups/data/ps1/PV3.4:/data/beegfs/astro-storage/groups/data/ps1/externals

    parser.add_argument('-a', '--area', type=str, required=True,
        help='string to define area: test for small area, stripe82 for stripe 82, full for whole area')

    parser.add_argument('-o', '--output', type=str, required=True,
        help='Name of the text/fits table with all the information')

    parser.add_argument('--wise_radius', type=float, default=2.0,  # consistant with thesis
        help='Search radius used to cross match with WISE (in arcsec)\
              (default: %(default)s) ')


    parser.add_argument('--nmax', type=int, default=1,
        help='Maximum number of neighbors to match\
              (default: %(default)s) ')


    parser.add_argument('--all', dest='allfields', action='store_true',
                        default=False,
                        help="Set to get all the fields of the requested\
                       databases.\
                       This could end in a very large table")
    parser.add_argument('--no-all', dest='allfields', action='store_false',
                         help="Set to get only some of the most important\
                       fields in the requested databases.\
                       This is hard-coded, talk to Eduardo to add more\
                       important fields.")


    parser.add_argument('--where', type=str, default='None',  #not working yet
        help='Optional parameter with a WHERE statement for the query\
        looking for objects in PS1 around the interesting objects.\
        Use When args.fast is True\
        If where=None (default) no where statement is applied.')

    parser.add_argument('--version', action='version', version='1.2')
    #based on eduardos code
    #version 1.1 query for ml_quasar
    #version 1.2 made query line up with bachelor thesis (no qf perfect and no primary det filter, sn for w2 band now works correctly, rest should be exactly the same)
    #version 1.3 added wise wise_designation and obj_id
    #version 1.4 added coadd_id to enable visual inspection
    return parser.parse_args()


if __name__ == '__main__':
    """Sample: python 1_query_catalog.py -o data/catalog/stripe82.fits -a "stripe82"
    python 1_query_catalog.py -a "test" -o data/catalog/test.fits"""

    print("version 1.4")

    start_time = time.time()
    args = parse_arguments()
    #lsd_db = os.getenv('LSD_DB')
    #lsd_db +=':' + args.dbpath


    #bounds for testing, when running on all data just set bounds to None
    if args.area == "full":
        decmin = -30
        decmax = 89.99999999
        ramin = -179.99999999
        ramax = 179.99999999
        bounds='rectangle({2:f}, {0:f}, {3:f}, {1:f})'.format(decmin, decmax, ramin, ramax)
    elif args.area == "stripe82":
        decmin = -1.26
        decmax = 1.26
        ramin = -60
        ramax = 60
        bounds='rectangle({2:f}, {0:f}, {3:f}, {1:f})'.format(decmin, decmax, ramin, ramax)
    elif args.area == "test":
        decmin = -1.26
        decmax = 1.26
        ramin = 1
        ramax = 1.001
        bounds='rectangle({2:f}, {0:f}, {3:f}, {1:f})'.format(decmin, decmax, ramin, ramax)
    else:
        #little larger testsize
        decmin = -1.26
        decmax = 1.26
        ramin = 1
        ramax = 1.5
        bounds='rectangle({2:f}, {0:f}, {3:f}, {1:f})'.format(decmin, decmax, ramin, ramax)
    print(bounds)

    ps1 = "PV34"

    #QUERY INITIALIZING
    if args.allfields:
        select ="*"

    select =''

    from_match_inner='{0:s}'.format(ps1)

    from_match_wise=', allwise(matchedto={0:s},nmax=1,dmax={1:f},inner)'.format(
                                        ps1, args.wise_radius)
    #from_match_wiserej=', allwiserej(matchedto={0:s},nmax=1,dmax={1:f},outer)'.format(
    #                                    args.dbname, args.wise_radius)




    ps1_important = get_ps1_important(ps1)

    if not args.allfields:
        select+=ps1_important
        select+=wise_important

    from_match_inner+=from_match_wise



    # if 'ebv' in args.catalog:
    #     select+=select_ebv


    where=args.where
                                                                    ##where is the error from??
    #(stack_primary_off==1) &

    #bands map as 0 -> g, 1-> r, 2-> i, 3-> z, 4-> y
    where = " (zPSFStackMag > 14) & (zPSFStackMag < 20.5) & (yPSFStackMag>0)" ##think about removing the faint limit!, zQf perfect missing, w1, w2 and w1 signal to noise of 5

    where += " & (mag_ap_stk.T[3] > 0) & ((glat > 20) | (glat< -20))" #checking that ap mag (mag_ap_stk.T[3]) exists to avoid errors with psf_ap
    where += " &((zPSFStackMag-mag_ap_stk.T[3]) < 0.3) & ((zPSFStackMag-mag_ap_stk.T[3]) > -0.3) "
    where +=" & (w1snr > 5) & (w2snr > 3)"
    #flags = objInfoFlag
    #will only use good 0x02000000 and good_stack 0x08000000 flags
    where += " & ((flags & 0x02000000)>0) & ((flags & 0x08000000)>0)"

    #also i want a blend flag for wise
    where += " & (na == 0) & (nb ==1)"
    #where += " & (stack_primary_off==1)"# does not work, mean and stack are already matched so makes not much sense anyways

    #np.warnings.filterwarnings('ignore') #we compare nan with smaller bigger, otherwise we get warnings, a lot of them
    #warnings.filterwarnings("ignore", message="RuntimeWarning: invalid value encountered in")
    #warnings.filterwarnings("error")
    lsd_query(from_db=from_match_inner,  bounds=bounds, select=select, output=args.output, where=where) #file format wrong!


    print("Elapsed Time: ", human_time(time.time() - start_time))

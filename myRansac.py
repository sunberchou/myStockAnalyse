# -*- coding: utf-8 -*-
import csv
import codecs
import time
import numpy
import re
from configparser import ConfigParser
import traceback


LOG_FILE = 'sz/l' + time.strftime( '%m%d.log', time.localtime())
BS_DATE = time.strptime( '2016/02/01', '%Y/%m/%d' )


def GetDataFromCSV( file_name = '' ):
    with codecs.open( file_name, 'r', 'gbk' ) as fp:
        reader = csv.reader( fp )
        row = reader.__next__()
        csv_name = row[0][7:11]
        pat = '\d{4}.\d{2}.\d{2}'
        data = []
        for row in reader:
            if None == re.match( pat, row[0] ) : continue
            if None == re.match( '\d+', row[4] ): continue
            new_d = time.strptime( row[0], '%Y/%m/%d' )
            if new_d < BS_DATE: continue
            b = float( row[4] )
            data.append( [row[0], b] )
    return data, csv_name


def GetCleanData( data_x, data_y ):
    i = 0
    j = 0
    data_out = []
    while i < len( data_x ) and j < len( data_y ):
        item_x = data_x[i]
        item_y = data_y[j]
        tx = time.strptime( item_x[0], '%Y/%m/%d' )
        ty = time.strptime( item_y[0], '%Y/%m/%d' )
        if tx < ty: 
            i += 1
            continue
        if tx > ty:
            j += 1
            continue
        data_out.append([item_x[1], item_y[1] ])
        i += 1
        j += 1
        #end of while loop
    return data_out


def log_msg( str = '' ):
    if str == '': return
    time_string = time.strftime( "%Y-%m-%d %X", time.localtime())
    with open( LOG_FILE,'a' ) as log_file:
        log_file.write( time_string + ': ' + str + '\r\n' )
    return


def ransac(data,model,n,k,t,d):
#fit model parameters to data using the RANSAC algorithm
#This implementation written from pseudocode found at
#http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
#Given:
#    data - a set of observed data points # 可观测数据点集
#    model - a model that can be fitted to data points #
#    n - the minimum number of data values required to fit the model
#    拟合模型所需的最小数据点数目
#    k - the maximum number of iterations allowed in the algorithm
#   最大允许迭代次数
#    t - a threshold value for determining when a data point fits a model
#   确认某一数据点是否符合模型的阈值
#    d - the number of close data values required to assert that a model fits well to data
#Return:
#    bestfit - model parameters which best fit the data (or nil if no good model is found)
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    best_d = d
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel,mayberesides = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = numpy.mean( better_errs )
            this_d = len(alsoinliers)
            if this_d > best_d:
                best_d = this_d
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        log_msg("did not meet fit acceptance criteria")
    
    return bestfit, {'inliers':best_inlier_idxs, 'lenth': best_d}

def random_partition(n,n_data):
#    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquaresModel:
#    """linear system solved using linear least squares
#    This class serves as an example that fulfills the model interface
#    needed by the ransac() function.
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = numpy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = numpy.dot(A,model)
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point

LOCAL_PATH = 'sz/'
BASE_FILE = 'SH#000001.txt'
FILE_SET = 'file_set.txt'
CONFIG_FILE = 'ransac.conf'

if __name__=='__main__':

    try:        #Get configurations form .config file
        config = ConfigParser()
        config.read( CONFIG_FILE )
        rs_n = config.getint( 'RANSAC', 'N_FIT' )
        rs_k = config.getint( 'RANSAC', 'MAX_ITR' )
        t_str = config.get( 'RANSAC', 'THRES' )
        rs_t = float( t_str )
        rs_d = config.getint( 'RANSAC', 'N_ALSO' )
        I_STR = config.get( 'RANSAC', 'I_CONST' )
        I_CONST = float( I_STR )

    except Exception as e:
        log_msg( 'configration file %s open fail.'%CONFIG_FILE )


    n_inputs = 1
    n_outputs = 1

    fx = LOCAL_PATH + BASE_FILE
    dataX, nameX = GetDataFromCSV( fx )
    f_set = LOCAL_PATH + FILE_SET
    file_list = []
    result = []
    with open( f_set, 'r' ) as fs:
        file_listln = fs.readlines( )
    pat = 'S[HZ]#\d{6}\.txt'
    for str in file_listln:
        m = re.match( pat, str )
        if None != m: file_list.append( m.group() )

    fw_name = LOCAL_PATH + 'ransac_result.txt'
    with open( fw_name, 'w' ) as fw_p:
        fw_p.write( '\r\n')
        
    for fn in file_list:
        File_Y = LOCAL_PATH + fn
        dataY, nameY = GetDataFromCSV( File_Y )
        dataXY = GetCleanData( dataX, dataY )
        all_data = numpy.array( dataXY )
        dx = all_data[:,0]
        mx = dx.mean()
        if mx == 0:
            log_msg( 'mean x is zero' )
            break;
        dx = (dx - mx )/mx
        dy = all_data[:,1]
        my = dy.mean()
        if my == 0:
            log_msg( 'mean y is zero' )
            continue
        dy = (dy - my)/my
        all_data = numpy.vstack(( dx, dy )).T
        input_columns = range(n_inputs) # the first columns of the array
        output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
        model = LinearLeastSquaresModel(input_columns,output_columns,debug=False)
        
        log_msg( 'Deal with %s.'%fn )
        # run RANSAC algorithm
        ransac_fit, ransac_data = ransac(
            all_data, model, rs_n, rs_k, rs_t, rs_d ) # misc. parameters

        if ransac_fit == None: continue
        ransac_value = ransac_fit[0,0]
        r_idx = re.match( 'S.#\d{6}', fn ).group()
        fw_name = LOCAL_PATH + 'o_' + r_idx + '.csv'
        item = [r_idx, nameY, ransac_value, ransac_data['lenth']]
        r_dta = float( 0 )
        with open( fw_name, 'w' ) as fw_p:
            for i in range( dx.size ):
                tmp = dy[i]-dx[i] * ransac_value
                r_dta = r_dta * ( 1-I_CONST ) + tmp * I_CONST
                fw_p.write( '%.6f, %.6f, %.6f, %.6f\r\n'%(
                    dx[i], dy[i], tmp, r_dta ))
        item.append( tmp )
        item.append( r_dta )
        fw_name = LOCAL_PATH + 'ransac_result.txt'
        with open( fw_name, 'a' ) as fw_p:
            fw_p.write( '%s, %s, %.6f, %d, %.6f, %.6f\r\n'%(
                item[0],item[1],item[2],item[3], item[4], item[5] ))
        #End to 'for' loop








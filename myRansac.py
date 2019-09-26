#!\usr\bin\env python3
#encoding: utf-8 
import csv
import codecs
import time
import numpy
import re
from configparser import ConfigParser
import traceback
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os


def GetDataFromCSV( file_name = '' ):
    with codecs.open( file_name, 'r', 'gbk' ) as fp:
        reader = csv.reader( fp )
        row = reader.__next__() #first line on top
        lstr = row[0].split()
        csv_name = lstr[1]
        if lstr[2] == '日线':
            idx = 4
        else:
            idx = 5
        pat = '\d{4}.\d{2}.\d{2}'
        data = []
        row = reader.__next__() #skip second line on top
        for row in reader:
            if len(row[0]) > 10:
                lstr = row[0].split()
            else:
                lstr = row
            if re.match( pat, lstr[0] ) is None: continue
            if re.match( '\d+', lstr[idx] ) is None: continue
            new_d = time.strptime( lstr[0], '%Y/%m/%d' )
            if new_d < BS_DATE: continue
            b = float( lstr[idx] )
            data.append( [lstr[0], b] )
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
def ransac(data,model,n,k,t,d):
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    best_d = d
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
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
        log_msg("Did not meet fit acceptance criteria")
    return bestfit, {'inliers':best_inlier_idxs, 'lenth': best_d}

#return n random rows of data (and also the other len(data)-n rows)
def random_partition(n,n_data):
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

#linear system solved using linear least squares
#This class serves as an example that fulfills the model interface
#needed by the ransac() function.
class LinearLeastSquaresModel:
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A0 = numpy.vstack([data[:,i] for i in self.input_columns])[0]
        A = numpy.vstack([A0, numpy.ones(len(A0))]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = numpy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        #B_fit = numpy.dot(A,model)
        B_fit = A * model[0][0] + model[1][0]
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point


class qqExmail:
    def __init__(self):
        self.user = 'zsb@cuteguide.cn'
        self.passwd = 'zhou111Qt'
        self.to_list = ['sunber.chou@qq.com']
        self.cc_list = ['zhousongbo@hanmingtech.com']
        self.tag = 'Finally, Ransac get result!'
        self.doc = None
        return
    def send(self):
        ret = True
        try:
            mail_host = smtplib.SMTP_SSL('smtp.exmail.qq.com', port=465)
            mail_host.login(self.user,self.passwd)
            receiver = self.to_list + self.cc_list
            mail_host.sendmail(self.user, receiver, self.get_attach())
            mail_host.close()
        except Exception as e:
            ret = False
        return ret
    def get_attach(self):
        message = MIMEMultipart()
        my_ip = os.popen('hostname -I').readlines()
        str = 'FYI: This mail is sent from a Ransac dev\r\n'
        str += 'Which IP addr is %s'%my_ip[0]
        txt = MIMEText(str)
        message.attach(txt)
        if self.tag is not None: 
            message['Subject'] = Header(self.tag,'utf-8')
        if self.user is not None: 
            message['From'] = Header('RansacDev<%s>'%self.user, 'utf-8')
        if len(self.to_list) > 0: 
            message['To'] = Header(';'.join(self.to_list), 'utf-8')
        if len(self.cc_list) > 0: 
            message['Cc'] = Header(';'.join(self.cc_list), 'utf-8')
        if self.doc: 
            fn = os.path.basename( self.doc )
            with open(self.doc,'rb') as f:
                doc = MIMEText(f.read(), 'base64', 'utf-8')
                doc["Content-Type"] = 'application/octet-stream'
                doc["Content-Disposition"] = 'attachment; filename="%s"'%fn
                message.attach(doc)
        return message.as_string()

if __name__=='__main__':

    CONFIG_FILE = 'ransac.conf'
    try:        #Get configurations form .config file
        config = ConfigParser()
        config.read( CONFIG_FILE )
        rs_n = config.getint( 'RANSAC', 'MIN_NUM' )
        rs_k = config.getint( 'RANSAC', 'MAX_ITR' )
        t_str = config.get( 'RANSAC', 'THRES' )
        rs_t = float( t_str )
        rs_d = config.getint( 'RANSAC', 'N_CLOSE' )
        I_STR = config.get( 'RANSAC', 'I_CONST' )
        I_CONST = float( I_STR )
        LOCAL_PATH = config.get( 'RANSAC', 'DATA_PATH' )
        BASE_FILE = config.get( 'RANSAC', 'BASE_FILE' )
        BASE_DATE = config.get( 'RANSAC', 'BASE_DATE' )
        BS_DATE = time.strptime( BASE_DATE, '%Y/%m/%d' )
    except Exception as e:
        exit(1)

    LOG_FILE = LOCAL_PATH + 'log' + time.strftime( '%y%m%d.log', time.localtime())
    n_inputs = 1
    n_outputs = 1

    fx = LOCAL_PATH + BASE_FILE
    dataX, nameX = GetDataFromCSV( fx )

    file_list = os.popen( 'ls %s*.txt'%LOCAL_PATH ).readlines()
    lstResult = []
    for fn in file_list:
        File_Y = fn.rstrip('\n')
        if File_Y == fx:
            continue
        dataY, nameY = GetDataFromCSV( File_Y )
        dataXY = GetCleanData( dataX, dataY )
        if len(dataXY) < 2*rs_d:
            continue
        all_data = numpy.array( dataXY )
        dx = all_data[:,0]
        mx = dx.mean()
        if mx == 0:
            log_msg( 'mean x is zero' )
            break
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

        log_msg( 'Deal with %s.'%File_Y )
        # run RANSAC algorithm
        ransac_fit, ransac_data = ransac(
            all_data, model, rs_n, rs_k, rs_t, rs_d ) # misc. parameters

        if ransac_fit is None: continue
        ransac_value = ransac_fit[0,0]
        ransac_rest = ransac_fit[1,0]
        r_idx = os.path.basename( File_Y )[ :-4]
        fnResult = LOCAL_PATH + 'o' + r_idx + '.csv'
        item = [r_idx, dx.size, nameY, ransac_value, ransac_rest, ransac_data['lenth']]
        r_dta = float( 0 )
        with open( fnResult, 'w' ) as fpResult:
            for i in range( dx.size ):
                tmp = dy[i]-dx[i] * ransac_value-ransac_rest
                r_dta = r_dta * ( 1-I_CONST ) + tmp * I_CONST
                fpResult.write( '%.6f, %.6f, %.6f, %.6f\r\n'%(
                    dx[i], dy[i], tmp, r_dta ))
        item.append( tmp )
        item.append( r_dta )
        lstResult.append( item )
        #End to 'for' loop
    lstResult.sort(key=lambda x:x[7], reverse = True )
    fnList = LOCAL_PATH + 'A_result.txt'
    with open( fnList, 'w', encoding='utf-8') as fw_p:
        fw_p.write( 'item[n], name, r_val, r_res, n_fit, f_v, f_dta\r\n')
        for item in lstResult:
            fw_p.write( '%s[%d], %s, %.6f, %.6f, %d, %.6f, %.6f\r\n'%(
                item[0], item[1],item[2],item[3], item[4], item[5], item[6], item[7] ))
    myMail = qqExmail()
    myMail.doc = fnList
    myMail.send()
    #end of file
    










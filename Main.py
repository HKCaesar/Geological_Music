"""Script for predicting geological location of music tracks
Final project for Stanford CME 193 Class
This script uses different learning models to predict
the latitude and longitude (geological location) of music tracks
based on its audio features
Refs:
[1] Zhou, Fang, Q. Claire, and Ross D. King. "Predicting the Geographical Origin of Music." Data Mining (ICDM), 2014 IEEE International Conference on. IEEE, 2014.
[2] https://archive.ics.uci.edu/ml/datasets/Geographical+Original+of+Music
by Shifan Mao 05/20/15
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
#import seaborn as sns
from mpl_toolkits.basemap import Basemap, addcyclic

import sklearn as sklearn
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

#PLOTTING OPTIONS: 1=plot, 0=hide plot
ploton1=1  #plot heatmap of features of first 200 tracks
ploton2=1  #plot gelogical distribution of tracks
ploton3=1  #plot training and test data with predictions from models

#INPUT FILE
#  containing features of track and geological location of each track
filename = 'data/features_track.txt'

#REGRESSION MODEL CHOICE
#AVAILABLE MODELS:
#model='Random Forest'   :: random forest model with n_estimators
#                           as number of trees in the forest
#                           default n_estimator=10
#model='SVM'             :: support vector matrix model with
#                           linear or polynomial kernel
#                           default kernel as linear
#model='Linar Regression':: linear regression model
#model='Linear Regression'
model='SVM'

#REGRESSION MODEL
def regression(model='Random Forest',n_estimators=10,kernel='linear'):
    if (model=='Random Forest'):
        regr = RandomForestRegressor(n_estimators=n_estimators)
    elif (model=='Linear Regression'):
        regr = linear_model.LinearRegression()
    elif (model=='SVM'):
        if (kernel=='linear'):
            regr = svm.SVR(kernel='linear',C=1.0)
        elif (kernel=='poly'):
            regr = svm.SVR(kernel='poly',degree=1)

    #Fit to longitude train data
    regr.fit(data_train,lons_train)
    #Predict training and test longitude data
    L0=regr.predict(data[0:N_train][:])
    Lp=regr.predict(data[N_train+1:NT-1][:])

    #Fit to latitude train data
    regr.fit(data_train,lats_train)
    #Predict training and test latitude data
    A0=regr.predict(data[0:N_train][:])
    Ap=regr.predict(data[N_train+1:NT-1][:])

    return(L0,Lp,A0,Ap)

#### OPEN FILE ####
lons = []  #longitude of each track
lats = []  #latitude of each track
data = []  #feature of each track

print "(1) Reading document ... "
with open(filename,'r') as f:
    for line in f:
        row = line.split(',')
        row[-1] = row[-1].strip()   #strip last element
        lons.append(float(row[-1])) #save to coordinates
        lats.append(float(row[-2]))
        data.append([float(i) for i in row[0:-3]])

NT=len(data)  #number of tracks (samples)
NF=len(data[0])  #number of features

#find number of distinct geographic origin locations of tracks
lons_unique=np.unique(lons)
NL=len(lons_unique)

print "  Number of tracks = ", NT
print "  Number of features = ", NF
print "  Number of distinct locations = ", NL

#### PLOT HEATMAP OF FEATURES ####
if ploton1:
    fig1=plt.figure(figsize=(4,8))
    #plt.imshow(data[0:60][:])
    #plt.imshow(data[:][:])
    plt.imshow(data[0:200][:])
    plt.hot()
    clbar=plt.colorbar()
    clbar.set_label('Normalized Feature Value')
    plt.grid()
    plt.xlabel('Feature #')
    plt.ylabel('Track #')
    plt.title('Figure 1: Audio Features of first 200 tracks')
    plt.show()

#### PLOT LATITUDE AND LONGITUDE ON MAP ####
if ploton2:
    fig2=plt.figure(figsize=(8,4.5))
    ax = fig2.add_axes([0.05,0.05,0.9,0.85])
    map = Basemap(projection='mill', resolution = 'l', area_thresh = 1000.0,
                  lat_0=0, lon_0=0)
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color = 'gray')
    map.drawparallels(np.arange(-80,81,20),labels=[1,1,0,0])
    map.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1])
    
    x,y = map(lons, lats)
    map.plot(x, y, 'ro', markersize=6) 
    ax.set_title('Figure 2: Distribution of 1059 tracks (one location can have many tracks)')
    plt.show()

#### TRAIN LEARNING MODEL WITH LINEAR KERNAL ####
# SPLIT INTO TRAINING AND TEST SETS
K=4
N_test=NT/K  #size of training set
N_train=NT-N_test

data_train = data[0:N_train][:]
lons_train = lons[0:N_train]
lats_train = lats[0:N_train]
print '(2) Training '+model+' Model ... '
[L0,Lp,A0,Ap] = regression(model=model)

# MEAN-SQUARED ERROR EVALUATION
L1=lons[0:N_train] #training data
Lt=lons[N_train+1:NT-1] #test data

A1=lats[0:N_train] #training data
At=lats[N_train+1:NT-1] #test data

LMSE_train=((L1-L0)**2).mean(axis=0)
AMSE_train=((A1-A0)**2).mean(axis=0)
print 'Longitude Training MSE=', LMSE_train
print 'Latitude Training MSE=', AMSE_train
LMSE_test=((Lt-Lp)**2).mean(axis=0)
AMSE_test=((At-Ap)**2).mean(axis=0)
print 'Longitude Test MSE=', LMSE_test
print 'Latitude Test MSE=', AMSE_test

if ploton3:
    #Training error :: Training data
    fig3=plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(L0,L1,'b.',label='model')
    plt.plot(L1,L1,'b-',linewidth=2.,label='x=y')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Training Lat. w/ ' + model + ' model')
    plt.xlim(-100,150)
    plt.ylim(-100,150)
    plt.legend(loc=2)

    plt.subplot(2,2,2)
    plt.plot(A0,A1,'r.',label='model')
    plt.plot(A1,A1,'r-',linewidth=2.,label='x=y')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Training Lon. w/ ' + model + ' model')
    plt.xlim(-40,60)
    plt.ylim(-40,60)
    plt.legend(loc=2)

    #Testing  error :: Compare with target function
    plt.subplot(2,2,3)
    plt.plot(Lp,Lt,'b.',label='model')
    plt.plot(Lt,Lt,'b-',linewidth=2.,label='x=y')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Test Lat. w/ ' + model + ' model')
    plt.xlim(-100,150)
    plt.ylim(-100,150)
    plt.legend(loc=2)

    plt.subplot(2,2,4)
    plt.plot(Ap,At,'r.',label='model')
    plt.plot(At,At,'r-',linewidth=2.,label='x=y')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Test Lon. w/ ' + model + ' model')
    plt.legend(loc=2)
    plt.xlim(-40,60)
    plt.ylim(-40,60)
    plt.show()

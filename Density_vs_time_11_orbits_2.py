import pandas as pd
from numpy import arange, exp, mean
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from glob import glob
import pdb
import os
from sys import exit

while 1:
    print "Would you like to plot multiple altitudes or one?"
    print "1. One Altitude"
    print "2. Multiple Altitudes"
    UserChoice = input()
    if UserChoice == 1:
        print "Enter Altitude"
        UserAlt = input()
        uLimit = UserAlt + 2.5
        lLimit = UserAlt - 2.5

        if UserAlt < 2.5:
            print "Invalid Altitude (Please select altitude >2.5km)"
            continue
        else:
            break
    elif UserChoice == 2:
        uLimit = 220
        lLimit = 160
        break
    else:
        print "Invalid selection.  Please enter 1 (single alt) or 2 (multiple alt)"
        continue





def read_one_file(file):

    f = open(file,'r')
    data = []
    next(f) #skip the file header line

    altlist = []
    # split across line, only keep data of interest

    for line in f:
        temp = line.split(',')
        datestring = temp[0]
        year = int(datestring[0:4])
        month = int(datestring[5:7])
        day = int(datestring[8:10])
        hour = int(datestring[11:13])
        minute = int(datestring[14:16])
        second = int(datestring[17:19])
        date = datetime(year,month,day,hour,minute,second)

        species = temp[13]
        alt = float(temp[7])
        lon = float(temp[9])
        lat = float(temp[10])
        orbit = float(temp[5])

        try:
            den = float(temp[15])
            altlist.append(alt)

            if alt <= uLimit and alt >= lLimit: # for this exercise, we only consider data below 300km
                tempdict = {'date': date, 'alt': alt, 'den': den, 'species': species, 'longitude': lon, 'latitude': lat, 'orbit': orbit}
                if alt <= altlist[-1]: # only want inbound leg of orbit
                    data.append(tempdict)
        except:
            pass
    return data

def read_ngims(mylist):
    alldata = []
    for file in mylist:
        temp_data = read_one_file(file)
        alldata.append(temp_data)

    return alldata

def date_range(startdate,enddate):
    filelist = glob('data/*cs*.csv') #grab all MAVEN data from downloaded files using glob
    if len(filelist) < 1:
        print "No MAVEN files found!"
        print "Exiting!"
        exit(1)
    mylist = []
    for file in filelist: # grab dates from file names
        fileyear = int(file[-27:-23])
        filemonth = int(file[-23:-21])
        fileday = int(file[-21:-19])
        filedate = datetime(fileyear,filemonth,fileday,0,0,0)

        if startdate <= filedate <= enddate:
            mylist.append(file)
    return mylist

startdate = datetime(2016,3,25)
enddate = datetime(2016,5,10)
#dates = pd.date_range(startdate,enddate,freq='2min')
mylist = date_range(startdate,enddate)
all_ngims = read_ngims(mylist)
all_ngims = [item for sublist in all_ngims for item in sublist]

ngims_df = pd.DataFrame.from_records(data = all_ngims, index = ('orbit'))









#generate list to prompt user of possible choices
print "Select Species"
print "1. Argon (Ar)"
print "2. Carbon Dioxide (CO2)"
print "3. Nitrogen Gas (N2)"
print "4. Helium (He)"
print "5. Carbon Monoxide (CO)"
print "6. Oxygen (O)"

#error check to make sure user selected a valid molecule
while 1:
    choice = raw_input()
    mydf = ngims_df[ngims_df['species'] == '%s' %(choice)]
    if mydf.empty:
        print "Error: No data for specified molecule -- Try again"
        continue
    else:
        break



#function to create x and y coordinates from associated altitude ranges
def create_data_points(mydataframe):
    mycoords = []
    mydataframe['date'] = np.int64(mydataframe['date'])
    mydataframe = mydataframe.groupby(mydataframe.index).mean()
    mydataframe_den_copy = mydataframe['den'].copy()
    mydataframe_date_copy = mydataframe['date'].copy()
    rolling_avg_den = mydataframe_den_copy.rolling(window = 11, center = True).mean()
    rolling_avg_date = mydataframe_date_copy.rolling(window =11, center = True).mean()
    x_values = rolling_avg_date.dropna()
    x_values = pd.DatetimeIndex(x_values)
    y_values = rolling_avg_den.dropna()
    mycoords.append(x_values)
    mycoords.append(y_values)
    return mycoords

##Logic Branch for Single Altitude

if(UserChoice == 1):
    #recast dates as 64 bit integers so they can be averaged
    mydf['date'] = np.int64(mydf['date'])
    #take repeated indices and condense them into one index, averaging the associated data
    mydf = mydf.groupby(mydf.index).mean()
    #make copies for pandas series to work with them
    df_den_copy = mydf['den'].copy()
    df_date_copy = mydf['date'].copy()
    #compute rolling averages for densities and dates
    rolling_avg_den = df_den_copy.rolling(window = 11, center = True).mean()
    rolling_avg_date = df_date_copy.rolling(window = 11, center = True).mean()
    #specify x-axis data (dates) and convert them back into Datetime format. Also drop rows containing empty value
    x_axis = rolling_avg_date.dropna()
    x_axis = pd.DatetimeIndex(x_axis)
    #specify y-axis data and remove rows containing an empty row
    y_axis = rolling_avg_den.dropna()
    #generate plot
    plt.plot(x_axis,y_axis, 'bo')
    #make dates look nicer
    plt.gcf().autofmt_xdate()
    #misc plot data
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('Density ($Molecules/cm^3$)')
    #plt.title('%i km'%(UserAlt))
    #show plot
    plt.show()

##Logic Branch for Multiple Altitudes
if(UserChoice == 2):
    #list of all x and y pairs
    allcoordinates = []
    #list of all Altitudes
    allaltitudes = []

    lower = 160
    dalt = 10
    altitude1 = mydf[(mydf['alt'] >= lower) & (mydf['alt'] < lower+dalt)]
    altitude2 = mydf[(mydf['alt'] >= lower+dalt) & (mydf['alt'] < lower+2*dalt)]
    altitude3 = mydf[(mydf['alt'] >= lower+2*dalt) & (mydf['alt'] < lower+3*dalt)]
    altitude4 = mydf[(mydf['alt'] >= lower+3*dalt) & (mydf['alt'] < lower+4*dalt)]
    altitude5 = mydf[(mydf['alt'] >= lower+4*dalt) & (mydf['alt'] < lower+5*dalt)]

    allaltitudes.append(altitude1)
    allaltitudes.append(altitude2)
    allaltitudes.append(altitude3)
    allaltitudes.append(altitude4)
    allaltitudes.append(altitude5)

    for x in allaltitudes:
        allcoordinates.append(create_data_points(x))


    plt.plot(allcoordinates[0][0], allcoordinates[0][1], 'ko', label='160-170 km')
    plt.plot(allcoordinates[1][0], allcoordinates[1][1], 'go', label='170-180 km')
    plt.plot(allcoordinates[2][0], allcoordinates[2][1], 'bo', label='180-190 km')
    plt.plot(allcoordinates[3][0], allcoordinates[3][1], 'co', label='190-200 km')
    plt.plot(allcoordinates[4][0], allcoordinates[4][1], 'ro', label='200-210 km')


    plt.legend(loc='best',fontsize = 9)
    plt.xlabel('Time',fontsize=18)
    plt.ylabel('Density ($Molecules/cm^3$)',fontsize=18)
    plt.gcf().autofmt_xdate()
    PearCorr = allcoordinates[0][1].corr(allcoordinates[1][1])
    print "The Pearson Correlation Coefficient between the 160-170 km and 170-180 km lines is: {}".format(PearCorr)
    plt.show()
pdb.set_trace()

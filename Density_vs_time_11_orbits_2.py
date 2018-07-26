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

startdate = datetime(2016,1,1)
enddate = datetime(2016,12,31)
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
    #rolling_avg_date = mydataframe_date_copy.rolling(window =11, center = True).mean()
    mydataframe = mydataframe.dropna()
    x_values = mydataframe['date']
    x_values = pd.DatetimeIndex(x_values)
    y_values = rolling_avg_den
    mycoords.append(x_values)
    mycoords.append(y_values)
    return mycoords

#function to determine a dataframe's rolling average and then reindex it by date
def date_indexed_rollavg(mydataframe):
    dateIndexDF = mydataframe.copy()
    dateIndexDF['date'] = pd.DatetimeIndex(dateIndexDF['date'])
    dateIndexDF['date'] = dateIndexDF['date'].dt.round('min')
    dateIndexDF['date'] = np.int64(dateIndexDF['date'])
    dateIndexDF = dateIndexDF.groupby(dateIndexDF.index).mean()
    dateIndexDF = dateIndexDF.rolling(window=11, center = True).mean()
    dateIndexDF = dateIndexDF.dropna()
    dateIndexDF['date'] = pd.DatetimeIndex(dateIndexDF['date'])
    dateIndexDF['date'] = dateIndexDF['date'].dt.round('min')
    dateIndexDF = dateIndexDF.set_index('date')
    #dateIndexDF.index = pd.DatetimeIndex(dateIndexDF.index)
    oidx = dateIndexDF.index
    nidx = pd.date_range(oidx.min(),oidx.max(),freq='270T')
    dateIndexDF = dateIndexDF.reindex(nidx,method='nearest',limit=1,tolerance='260Min').interpolate(limit=1)

    #extract the density series from the DataFrame
    dateIndexDS = dateIndexDF['den']
    return dateIndexDS

#function to determine the correlation between two density dataseries
def density_pCorrCoef(dataSeriesOne, dataSeriesTwo):
    correlation = pd.rolling_corr(dataSeriesOne, dataSeriesTwo, window = 16, min_periods=10, center = True)
    return correlation.dropna()

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
    plt.xlabel('Time',fontsize=18)
    plt.ylabel('Density ($Molecules/cm^3$)',fontsize=18)
    #plt.title('%i km'%(UserAlt))
    #show plot
    plt.show()

##Logic Branch for Multiple Altitudes
if(UserChoice == 2):
    #list of all x and y pairs
    allcoordinates = []
    #list of all Altitudes
    allaltitudes = []
    #specify lower alt(km) and size of alt bin
    lower = 160
    dalt = 10
    altitude1 = mydf[(mydf['alt'] >= lower) & (mydf['alt'] < lower+dalt)]
    altitude1 = altitude1.groupby(altitude1.index).filter(lambda x: ((x['alt'] < lower+1) & (x['alt'] > lower)).any())
    altitude1 = altitude1.groupby(altitude1.index).filter(lambda x: ((x['alt'] > lower+9) & (x['alt'] < lower +10)).any())

    altitude2 = mydf[(mydf['alt'] >= lower+dalt) & (mydf['alt'] < lower+2*dalt)]
    altitude2 = altitude2.groupby(altitude2.index).filter(lambda x: ((x['alt'] < lower+11) & (x['alt'] > lower+10)).any())
    altitude2 = altitude2.groupby(altitude2.index).filter(lambda x: ((x['alt'] > lower+19) & (x['alt'] < lower +20)).any())

    altitude3 = mydf[(mydf['alt'] >= lower+2*dalt) & (mydf['alt'] < lower+3*dalt)]
    altitude3 = altitude3.groupby(altitude3.index).filter(lambda x: ((x['alt'] < lower+21) & (x['alt'] > lower+20)).any())
    altitude3 = altitude3.groupby(altitude3.index).filter(lambda x: ((x['alt'] > lower+29) & (x['alt'] < lower +30)).any())

    altitude4 = mydf[(mydf['alt'] >= lower+3*dalt) & (mydf['alt'] < lower+4*dalt)]
    altitude4 = altitude4.groupby(altitude4.index).filter(lambda x: ((x['alt'] < lower+31) & (x['alt'] > lower+30)).any())
    altitude4 = altitude4.groupby(altitude4.index).filter(lambda x: ((x['alt'] > lower+39) & (x['alt'] < lower +40)).any())

    altitude5 = mydf[(mydf['alt'] >= lower+4*dalt) & (mydf['alt'] < lower+5*dalt)]
    altitude5 = altitude5.groupby(altitude5.index).filter(lambda x: ((x['alt'] < lower+41) & (x['alt'] > lower+40)).any())
    altitude5 = altitude5.groupby(altitude5.index).filter(lambda x: ((x['alt'] > lower+49) & (x['alt'] < lower +50)).any())

    allaltitudes.append(altitude1)
    allaltitudes.append(altitude2)
    allaltitudes.append(altitude3)
    allaltitudes.append(altitude4)
    allaltitudes.append(altitude5)

    for x in allaltitudes:
        allcoordinates.append(create_data_points(x))

    plt.figure(1)
    plt.plot(allcoordinates[0][0], allcoordinates[0][1], 'ko', label='{}-{} km'.format(lower,lower+dalt))
    plt.plot(allcoordinates[1][0], allcoordinates[1][1], 'go', label='{}-{} km'.format(lower+dalt,lower+2*dalt))
    plt.plot(allcoordinates[2][0], allcoordinates[2][1], 'bo', label='{}-{} km'.format(lower+2*dalt,lower+3*dalt))
    plt.plot(allcoordinates[3][0], allcoordinates[3][1], 'co', label='{}-{} km'.format(lower+3*dalt,lower+4*dalt))
    plt.plot(allcoordinates[4][0], allcoordinates[4][1], 'ro', label='{}-{} km'.format(lower+4*dalt,lower+5*dalt))


    plt.legend(loc='best',fontsize = 9)
    plt.xlabel('Time',fontsize=18)
    plt.ylabel('Density ($Molecules/cm^3$)',fontsize=18)
    plt.gcf().autofmt_xdate()
    PearCorr = allcoordinates[0][1].corr(allcoordinates[1][1])
    print "The average Pearson Correlation Coefficient between the 160-170 km and 170-180 km lines is: {}".format(PearCorr)

    pearCorrCoef = density_pCorrCoef(date_indexed_rollavg(altitude2), date_indexed_rollavg(altitude3))
    plt.figure(2)
    plt.xlabel('Time',fontsize = 18)
    plt.ylabel('Pearson Correlation Coefficient',fontsize=18)
    plt.title('Pearson Correlation Coefficient versus time')
    plt.gcf().autofmt_xdate()
    plt.plot(pearCorrCoef.index, pearCorrCoef,'b-')
    plt.show()
pdb.set_trace()


####How to recreate minutes between data point plot
#Emsure bin nhas been grouped by orbit number
#Convert dates from np.int64 to DatetimeIndex and round to nearest minute using df['date'].dt.round('min')
#Create new column in an altitude bin (e.g. altitude2) which represents the difference between successive rows in said DataFrame
#this is done with SampleAlttudeBin['difference']=SampleAlttudeBin['date'].sub[SampleAlttudeBin['date'].shift()).fillna(0)
#Recast SampleAlttudeBin['difference'].astype('timedelta64[m]')
#Plot!

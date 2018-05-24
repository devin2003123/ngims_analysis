import pandas as pd
from numpy import arange, exp, mean
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from glob import glob
import pdb
import os
from sys import exit

while -1:
    print "Enter Altitude"
    UserAlt = input()

    if UserAlt < 2.5:
        print "Invalid Altitude (Please select altitude >2.5km)"
        continue
    else:
        break

uLimit = UserAlt + 2.5
lLimit = UserAlt - 2.5


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



#plot the graph with chosen element

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
plt.title('%i km'%(UserAlt))
pdb.set_trace()
#show plot
plt.show()

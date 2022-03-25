# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 01:21:02 2022

@author: Andy
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def getPlot(dataCol, timeCol, dfResults, title, xx = None, yy = None):
    """
    Function to create plots for each of the isolation forest models. It creates a line plot of the original data, then creates a scatter plot of the anomalies on top.
    
    Parameters
    ----------
    dataCol : Array of int
        The data to fit to isolation forest and to create features for.
    timeCol : Either a datetime column or a datetimeindex
        The time element pair with the data
    dfResults : Pandas dataframe
        The dataframe passed by other functions that contains the isolation forest results
    title : String
        The pre-set title for each of the plots to describe which features were used in the model. 
    xx : String, optional
        The optional string for the x axis. The default is None.
    yy : String, optional
        The optional string for the y axis. The default is None.

    Returns
    -------
    A plot of the results.

    """
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Selects the data from the results dataframe that's flagged as anomalous.
    a = dfResults.loc[dfResults['Anomaly'] == -1, ['Date', 'Incidents']] #anomaly
    # Line plot of the base default values.
    ax.plot(timeCol, dataCol, color='C0', label='Normal')
    # Makes a scatter plot in the same frame as the line plot so that each of the anomalous data appears as a red dot.
    ax.scatter(a['Date'],a['Incidents'], color='red', label='Anomaly')
    plt.xlabel(xx)
    plt.ylabel(yy)
    plt.title(title)
    plt.legend()
    plt.show()

def getBase(dataCol, timeCol, xx = None, yy = None, time = None):
    """
    Parameters
    ----------
    dataCol : Array of int
        The data to fit to isolation forest and to create features for.
    timeCol : Either a datetime column or a datetimeindex
        The time element pair with the data
    xx : String, optional
        The optional string for the x axis. The default is None.
    yy : String, optional
        The optional string for the y axis. The default is None.
    time : Boolean, optional
        If set to True the end result dataframe will contain a datetime column. If None the result will contain only the date. The default is None.

    Returns
    -------
    A dataframe of the output of the isolation forest fitted to the base values and a plot of the results.

    """
    clf=IsolationForest(random_state=42)
    
    clf.fit(dataCol.to_numpy().reshape(-1,1))
    
    # Creating the DF with the results:
    #   Scores is the decimal scoring system that isolation forest uses
    #   The anomaly column is to simplify looking up values that are anomalous or not
    #   The default return values for anomaly are -1 for an anomaly and 1 for non-anomalous
    
    dfResults = pd.DataFrame()
    dfResults['baseScores']=clf.decision_function(dataCol.to_numpy().reshape(-1,1))
    dfResults['Anomaly']=clf.predict(dataCol.to_numpy().reshape(-1,1))
    dfResults['Incidents'] = dataCol.values
    
    # Doing these two seperately because a datetimeindex can't be converted with dt, so it is turned into a datetime then date.
    # If the data type was only input as a column of datetime this wouldn't be required, however this gives the option of using a datetimeindex.
    # time == none is to include only date, time == true is to include the time if its included in the input
    
    if time == None:
        dfResults['Date'] = timeCol
        dfResults['Date'] = dfResults['Date'].dt.date
    elif time == True:
        dfResults['Date'] = timeCol

    
    getPlot(dataCol, timeCol, dfResults, xx = xx, yy = yy, title = "Raw Input")
    
    return(dfResults)

def getBaseDelta(dataCol, timeCol, xx = None, yy = None):
    """
    Parameters
    ----------
    dataCol : Array of int
        The data to fit to isolation forest and to create features for.
    timeCol : Either a datetime column or a datetimeindex
        The time element pair with the data
    xx : String, optional
        The optional string for the x axis. The default is None.
    yy : String, optional
        The optional string for the y axis. The default is None.

    Returns
    -------
    A dataframe of the output of the isolation forest fitted to both the base value and a value for the difference from the prior measurement, also returns a plot of the results.

    """
    clf=IsolationForest(random_state=42)
    
    deltaCol = dataCol.diff()
    deltaPD = pd.concat([deltaCol,dataCol], axis = 1)
    deltaPD.columns = ("deltaCol", "dataCol")
    
    # When using the difference between rows it will create an na value for the first row, I'm using the below method to do so,
    # Combined with the row creating deltaIncidents and the deltaDate
    # For the difference alone I could hard code the 1 value
    # However, the dataframes for the windows requires a more robust method and I would prefer to use that when possible over hard coding numbers

    #dropped = deltaPD['deltaCol'].isna().sum()    
    deltaPD = deltaPD.dropna()

    clf.fit(deltaPD)
    dfResults = pd.DataFrame()
    dfResults['deltaScores']=clf.decision_function(deltaPD)
    dfResults['Anomaly']=clf.predict(deltaPD)
    
    # When using the difference between rows it will create an na value for the first row, below is making the length of the arrays match.
    
    dfResults['Incidents'] = dataCol[1:].values
    dfResults['deltaIncidents'] = deltaCol[1:].values 
    dfResults['Date'] = timeCol[1:]

    getPlot(dataCol, timeCol, dfResults, xx = xx, yy = yy, title = "Raw Input and Delta")
    
    # Dropping the date so that the DF of the output only 1 date will be present, using the date column from the initial model for that.
    dfResults = dfResults.drop('Date', 1)
    dfResults = dfResults.drop('Incidents', 1)
    dfResults.columns.values[1] = "deltaAnomaly"
    return(dfResults)


def getWindows(dataCol, timeCol, windowList, xx = None, yy = None):
    """
    Parameters
    ----------
    dataCol : Array of int
        The data to fit to isolation forest and to create features for.
    timeCol : Either a datetime column or a datetimeindex
        The time element pair with the data
    windowList : List of ints
        The size of the rolling window averages to model.
    xx : String, optional
        The optional string for the x axis. The default is None.
    yy : String, optional
        The optional string for the y axis. The default is None.

    Returns
    -------
    A dataframe of the output(s) of the isolation forest fitted to the selected rolling window sizes, also returns a plot of the results.
    A dataframe of the rolling window values to be used in the getWindowsDelta function.

    """
    clf=IsolationForest(random_state=42)
    
    # windowresults is used for named columns in the returned endresults, while rollingdf is used for the isolation forest with the delta in the next function.
    
    rollingDF = pd.DataFrame()
    windowResults = pd.DataFrame()
    
    for val in windowList:
        rollingCol = dataCol.rolling(val).mean()
        dropped = rollingCol.isna().sum()
        rollingCol = rollingCol.dropna()
        
        clf.fit(rollingCol.to_numpy().reshape(-1,1))
    
    
        dfResults = pd.DataFrame()
        dfResults['scores']=clf.decision_function(rollingCol.to_numpy().reshape(-1,1))
        dfResults['Anomaly']=clf.predict(rollingCol.to_numpy().reshape(-1,1))
        dfResults['windowIncidents'] = rollingCol.values
        dfResults['Incidents'] = dataCol[dropped:].values
        dfResults['Date'] = timeCol[dropped:]
        
        getPlot(dataCol, timeCol, dfResults, xx = xx, yy = yy, title = ('Window of ' + str(val)))
        
        dfResults = dfResults.drop('Date', 1)
        dfResults = dfResults.drop('Incidents', 1)
        rollingDF = pd.concat([rollingDF,rollingCol], axis=1)
        
        dfResults.columns = ['window'+str(val)+'Scores', 'window'+str(val)+'Anomaly','window'+str(val)+'Incidents']
        windowResults = pd.concat([windowResults,dfResults],axis=1)
        
    return rollingDF, windowResults

def getWindowsDelta(dataCol, timeCol, windowList, rollingDF, xx = None, yy = None):
    """
    

    Parameters
    ----------
    dataCol : Array of int
        The data to fit to isolation forest and to create features for.
    timeCol : Either a datetime column or a datetimeindex
        The time element pair with the data
    windowList : List of ints
        The size of the rolling window averages to model.
    rollingDF : Pandas dataframe
        A column for the values of each of the rolling windows calculated from the dataCol in the getWindows function.
    xx : String, optional
        The optional string for the x axis. The default is None.
    yy : String, optional
        The optional string for the y axis. The default is None.

    Returns
    -------
    A dataframe of the output(s) of the isolation forest fitted to the selected rolling window sizes and a value for the difference from the prior measurement, also returns a plot of the results.

    """
    
    clf=IsolationForest(random_state=42)
    
    windowDeltaResults = pd.DataFrame()
    
    # windowList is used in this function during the plotting
    
    for val, (columnName, columnData) in zip(windowList,rollingDF.iloc[:,:].iteritems()):
        deltaCol = dataCol.diff()
        deltaPD = pd.concat([deltaCol,dataCol], axis = 1)
        deltaPD.columns = ("deltaCol", "dataCol")
        dropped = deltaPD['deltaCol'].isna().sum()
        deltaPD = deltaPD.dropna()
        
        clf.fit(deltaPD)
        
        dfResults = pd.DataFrame()
        dfResults['scores']=clf.decision_function(deltaPD)
        dfResults['Anomaly']=clf.predict(deltaPD)  
        dfResults['Incidents'] = dataCol[dropped:].values
        dfResults['deltaIncidents'] = deltaCol[dropped:].values
        dfResults['Date'] = timeCol[dropped:]
        

        getPlot(dataCol, timeCol, dfResults, xx = xx, yy = yy, title = ('Window of ' + str(val) + ' and Delta'))
        
        dfResults = dfResults.drop('Date', 1)
        dfResults = dfResults.drop('Incidents', 1)

        dfResults.columns = ['windowDelta'+str(val)+'Scores', 'windowDelta'+str(val)+'Anomaly','windowDelta'+str(val)+'Incidents']
        windowDeltaResults = pd.concat([windowDeltaResults,dfResults],axis=1)
        
    return windowDeltaResults

def getIF(dataCol, timeCol, windowList, x = None, y = None, time = None):
    """
    Parameters
    ----------
   dataCol : Array of int
        The data to fit to isolation forest and to create features for.
    timeCol : Either a datetime column or a datetimeindex
        The time element pair with the data
    windowList : List of ints
        The size of the rolling window averages to model.
    x : String, optional
        The optional string for the x axis. The default is None.
    y : String, optional
        The optional string for the y axis. The default is None.
    time : Boolean, optional
        If set to True the end result dataframe will contain a datetime column. If None the result will contain only the date. The default is None.
        
    Returns
    -------
    isoResults : Pandas dataframe
        Contains the results for isolation forest models fitted to each of the following:
            the base values
            the base value and a value for the difference from the prior measurement
            the selected rolling window sizes of the base values
            the selected rolling window sizes of the base values and a value for the difference from the prior measurement
        The dataframe also contains a column for the time component
    A plot highlighting the anomalies against the base values for each of the models fitted in the results dataframe.
            
    """
    assert isinstance(dataCol, pd.core.series.Series), 'dataCol should be a pandas series'
    assert isinstance(timeCol, pd.core.indexes.datetimes.DatetimeIndex), 'timeCol should be a pandas DatetimeIndex'
    assert isinstance(windowList, list), 'windowList should be a list'
    baseDF = getBase(dataCol, timeCol, xx = x, yy = y, time = time)
    rollingCol, rollingDF = getWindows(dataCol, timeCol, windowList, xx = x, yy = y)
    baseDeltaDF = getBaseDelta(dataCol, timeCol, xx = x, yy = y)
    windowDeltaDF = getWindowsDelta(dataCol, timeCol, windowList, rollingCol, xx = x, yy = y)
    
    isoResults = pd.concat([baseDF, baseDeltaDF, rollingDF, windowDeltaDF], axis = 1)
    
    # When concatenating the dataframes, since they are unequal length, there would be nan values at the bottom of every column except the base values.
    # The loop below is counting the number of nan values, dropping the nan values, then shifting the column down equal to the number of nan values.
    # This process would create the nan values at the top, and this is important so all of the values from the various data frames line up with the proper date time.
    # This step could be avoided by not dropping nan values before fitting the isolation forest, however, the IsolationForest function does not allow nan inputs.
    
    for (columnName, columnData) in isoResults.iloc[:,:].iteritems():
        dropped = columnData.isna().sum()
        test = columnData.dropna()
        isoResults[columnName] = test
        isoResults[columnName] = isoResults[columnName].shift(dropped, axis = 0)
     
    return isoResults


#%%
theList = [2,3]

oneDate = getIF(dfDay['Total_Incidents'], dfDay.index, theList, x = "dateTime", y = "timeSeriesData", time = True)
#%%
Do:
    replace plotting stuff with a function
    
Questions to ask:
    Do I explain why I do things yes
    For passing an argument that seems unneeded but is actually used for plotting, should I point that out? (almost forgot myself) yes
    Should a readme explaining why I did the project include I? if needed but avoid
    
#%%

Do:
    Anomaly names across each of the functions need to be standardized
    It skipped get baseDelta
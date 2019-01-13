import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import preprocessing, feature_selection, svm, gaussian_process
from sklearn import metrics, naive_bayes
from sklearn import neighbors, datasets, linear_model, ensemble, model_selection
from stockstats import StockDataFrame
import copy
import time
from datetime import date,timedelta
import warnings
warnings.filterwarnings("ignore")


#python script
import iex_collect as iex

def preprocess(data,indicators):
    pd.set_option('max_columns',10)
    stock_df = StockDataFrame.retype(data)
    for i in indicators:
        stock_df[i]
    data=data[['close',*indicators]]

    #create direction column
    direction = data['close'].pct_change()

    quarterDev=pd.DataFrame.std(direction)*0.4
    bins=[-1,-1*quarterDev,quarterDev,1]
    data['direction'] = pd.cut(direction,bins=bins,labels=[-1,0,1]).to_frame()

    data['updown']=pd.cut(direction,bins=[-1,0,1],labels=[-1,1]).to_frame()
    data[['direction','updown']]=data[['direction','updown']].shift(-1)

    current=data.tail(1)

    #missing data and first ten deleted
    data=data.replace([np.inf], np.nan).dropna()
    data=data.iloc[10:]

    print(data.tail())
    exit()

    #Normalize Data
    MinMax=preprocessing.MinMaxScaler()
    data[[*indicators]]=MinMax.fit_transform(data[[*indicators]])

    return data[['close',*indicators,'direction','updown']].dropna()

############################################################

def findBestFeatures(data,features,k):
    X=data[[*features]]
    y=data['direction']
    selectClf = feature_selection.SelectKBest(feature_selection.chi2,k=k)
    selectClf.fit(X,y)
    newX=X.columns[selectClf.get_support(indices=True)]
    return list(newX)

def customAccuracy(true,prediction):
    combined=pd.DataFrame(true)
    combined['prediction']=prediction

    combined=combined.replace([0],np.nan).dropna()
    total=combined.shape
    acc= metrics.accuracy_score(combined['updown'],combined['prediction'])
    return acc, total[0]

def gridSearch(features,tests,paramGrid,data):
    dfdict={'test':[],'accuracy':[]}

    X=data[features]
    y=data['direction']
    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.40,shuffle=False)
    XValidate,XTest,yValidate,yTest = model_selection.train_test_split(XTest,yTest,test_size=0.50,shuffle=False)

    # values of up down corresponding to yValidate (yValidate with 0 filled in)
    yTrue=data['updown'].loc[yValidate.index.tolist()]

    for test in tests:
        i=tests.index(test)
        testParam=paramGrid[i]
        for paramName in testParam:
            for c in testParam[paramName]:
                setattr(test,paramName,c)
                test.fit(XTrain,yTrain)
                yPredict=test.predict(XValidate)

                acc, total =customAccuracy(yTrue,yPredict)

                if total>=30:
                    dfdict['accuracy'].append(acc)
                    dfdict['test'].append(copy.deepcopy(test))

    df=pd.DataFrame(dfdict)
    return df

def findBestModel(df,data,featList):
    best_validation_score=df['accuracy'].max()

    # find bad model or non working model
    if best_validation_score<.50:
        return 'model not robust enough'
    try: best_index=df.index[df['accuracy']==best_validation_score].tolist()[0]
    except: return 'model inconclusive'

    y=data['direction']
    X=data[[*featList]]
    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.20,shuffle=False)

    # values of up down corresponding to yValidate (yValidate with 0 filled in)
    yTrue=data['updown'].loc[yTest.index.tolist()]

    # find what the best model is and test with 80%
    best_series=df.iloc[best_index]
    best_test=best_series['test']
    best_test.fit(XTrain,yTrain)
    yPredict=best_test.predict(XTest)
    best_test_score, tested =customAccuracy(yTrue,yPredict)
    total=yTest.shape[0]

    print('best test : ' + str(best_test))
    print('validation score : ' + str(best_validation_score))
    print('test score : ' + str(best_test_score))
    print('tested: %d / %d' % (tested,total))
    return(best_test_score,tested, total, best_test)


############################################################

def prediction(clf,features,data):
    X=data[features]
    y=data['direction']
    clf.fit(X,y)
    print(X.tail(1))
    action=clf.predict(X.tail(1))
    if action==1: action='price is predicted to rise in the next trading day'
    elif action==-1: action='price is predicted to fall in the next trading day'
    elif action==0: action='movement is not foreseeable'
    return action

def addNew(data):
    new_data=iex.stock_info(stock).return_today_data()
    new_data[0]['close']=new_data[0]['delayedPrice']
    data=data.append(new_data)
    return data

###############################################################################

def execution(stock,predict=False):
    featList=['rsi_14','adx','macd','open_2_sma','wr_10','cci','dma','trix','vr']

    clfList=[#neighbors.KNeighborsClassifier(),
            #gaussian_process.GaussianProcessClassifier(),
            #ensemble.RandomForestClassifier(),
            #naive_bayes.GaussianNB(),
            linear_model.LogisticRegression()]

    gridList=[#{'n_neighbors': [n for n in range(1,30,2)]},
            #{'kernel':[None]},
            #{'n_estimators': [n for n in range(1,100, 5)]},
            #{'priors':[None]},
            {'C': [n for n in np.arange(0.1,1.1,0.1)]}]

    data=iex.stock_info(stock).create_ranged_dataset('5y')
    data=addNew(data)

    data=preprocess(data,featList)
    if data.shape[0]<900:
        return 'dataset too small'

    features=findBestFeatures(data,featList,5)
    print(features)

    df=gridSearch(features,clfList,gridList,data)
    model=findBestModel(df,data,features)
    if type(model)==str: 
        return model

    acc,tested,total,clf=model

    if predict==True:
        return acc,features,tested,total,clf, prediction(clf,features,data)

    return (acc,features,tested,total,clf)


###########################################################

if __name__=='__main__':
    stockList=['baba','wmt','aapl','goog','amzn','orcl','fb','twtr',
                'cmg','gis','k','mcd','hsy','tsn','spy']
    stockList=['ba','vz','ibm','bidu','spy','aaba',
                'pnc','gs','bac','wfc']
    #stockList=['aaba','jpm','td','vod','qcom','cmcsa','cost','amgn',
    #            'eric','ebay','celg']

    totalSuccess,totalTested,totalObserved=0,0,0

    divisor=len(stockList)
    for stock in stockList:
        print(stock)
        result=execution(stock,predict=True)
        if type(result)==str:
            print(result)
            print('---')
            divisor-=1
            continue
        acc,features,tested,observed,clf,forecast=result
        print(forecast)

        if tested!=0:
            totalTested+=tested
            totalSuccess+=acc*tested
        totalObserved+=observed
        print('---')
    print('average accuracy: %.8f' % (float(totalSuccess)/float(totalTested)))
    print('test / observed = %d / %d = %.8f' % (totalTested,totalObserved,float(totalTested)/totalObserved))

# Success rate of Model (label incomplete vs insufficient model)
# calculate profit and loss from testing set
# Create Prediction function
# print confusion matrix
# Create Visual
# test if combining data from all companies is good


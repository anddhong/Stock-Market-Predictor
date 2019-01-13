import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import preprocessing, feature_selection, svm, gaussian_process
from sklearn import metrics, naive_bayes
from sklearn import neighbors, datasets, linear_model, ensemble, model_selection
from stockstats import StockDataFrame
import copy
import warnings
warnings.filterwarnings("ignore")
pd.set_option('max_columns',10)


#python script
import iex_collect as iex

def preprocess(data,indicators):
    stock_df = StockDataFrame.retype(data)
    for i in indicators:
        stock_df[i]
    data=data[['close',*indicators]]

    #create direction column
    direction = data['close'].pct_change()
    bins=[-1,0,1]
    data['direction'] = pd.cut(direction,bins,labels=[-1,1]).to_frame()
    data['direction']=data['direction'].shift(-1)
    data['rsi_change'] = data['rsi_14'].pct_change()

    data=data.replace([np.inf], np.nan).replace([0],np.nan).dropna()

    #Normalize Data
    MinMax=preprocessing.MinMaxScaler()
    data[['rsi_change',*indicators]]=MinMax.fit_transform(data[['rsi_change',*indicators]])

    return data[['close',*indicators,'direction','rsi_change']]

############################################################

def findBestParams(data,features,k):
    X=data[[*features]]
    y=data['direction']
    selectClf = feature_selection.SelectKBest(feature_selection.chi2,k=k)
    selectClf.fit(X,y)
    newX=X.columns[selectClf.get_support(indices=True)]
    return list(newX)

def gridSearch(features,tests,paramGrid,data):
    dfdict={'test':[],'accuracy':[]}
    y=data['direction']

    # best model with validation
    print('...')
    X=data[features]
    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.40,shuffle=False)
    XValidate,XTest,yValidate,yTest = model_selection.train_test_split(XTest,yTest,test_size=0.50,shuffle=False)
    for test in tests:
        i=tests.index(test)
        testParam=paramGrid[i]
        for paramName in testParam:
            for c in testParam[paramName]:
                setattr(test,paramName,c)
                test.fit(XTrain,yTrain)
                yPredict=test.predict(XValidate)
                acc=metrics.accuracy_score(yValidate,yPredict)

                dfdict['accuracy'].append(acc)
                dfdict['test'].append(copy.deepcopy(test))
                #deepcopy prevents changing dataframe

    df=pd.DataFrame(dfdict)
    return df

def findBestModel(df,data,featList):
    best_validation_score=df['accuracy'].max()
    if best_validation_score<.50:
        return 'model not robust enough'

    best_index=df.index[df['accuracy']==best_validation_score].tolist()[0]

    ##########
    best_series=df.iloc[best_index]
    best_test=best_series['test']

    y=data['direction']
    X=data[[*featList]]

    '''
    # best model with 60% training set
    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.40,shuffle=False)
    XValidate,XTest,yValidate,yTest = model_selection.train_test_split(XTest,yTest,test_size=0.50,shuffle=False)
    '''

    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.20,shuffle=False)

    # best model with 80% training set
    best_test.fit(XTrain,yTrain)
    yPredict=best_test.predict(XTest)
    best_test_score=metrics.accuracy_score(yTest,yPredict)

    print('best test : ' + str(best_test))
    print('validation score : ' + str(best_validation_score))
    print('test score : ' + str(best_test_score))

    return best_test_score,best_test


############################################################

def execution(stock):
    featList=['rsi_14','adx','macd','open_2_sma','wr_10','cci','dma','trix','vr']

    clfList=[#neighbors.KNeighborsClassifier(),
            #gaussian_process.GaussianProcessClassifier(),]
            #ensemble.RandomForestClassifier(),]
            #naive_bayes.GaussianNB(),
            linear_model.LogisticRegression()]

    gridList=[#{'n_neighbors': [n for n in range(1,30,2)]},
                #'weights':['uniform','distance']},
            #{'kernel':[None]},]
            #{'n_estimators': [n for n in range(1,200, 5)]},]
            #{'priors':[None]},
            {'C': [n for n in np.arange(0.1,1.1,0.1)]}]

    data=iex.stock_info(stock).create_ranged_dataset('5y')
    data=preprocess(data,featList)

    print('shape: ' + str(data.shape))
    if data.shape[0]<1000:
        return 'dataset too small'

    features=findBestParams(data,featList,5)
    print(features)

    df=gridSearch(features,clfList,gridList,data)
    model=findBestModel(df,data,features)
    if type(model)==str:
        return model

    acc=model[0]
    best_test=model[1]

    print('---')
    return(acc,features,best_test)

def modelPrediction(stock):
    try:
        acc, features, best_test = execution(stock)
    except:
        return execution(stock)



###########################################################

if __name__=='__main__':
    stockList=['baba','wmt','aapl','goog','amzn','orcl','fb','twtr',
                'cmg','gis','k','khc','mcd','hsy','tsn','spy']
    #stockList=['ba','roku','vz','ibm','bidu','spy','bgne','aaba',
    #            'pnc','gs','bac','wfc']

    meanAcc=0
    divisor=len(stockList)
    for stock in stockList:
        print(stock)
        result=execution(stock)
        if type(result)==str:
            print(result)
            divisor-=1
            continue
        acc=result[0]
        meanAcc+=acc
    print('average accuracy: %.8f' % (meanAcc/float(divisor)))


# print confusion matrix
# fix gridSearch
# test if combining data from all companies is good
# try probability threshold
# Why is there Close in featList
# Success rate of Model
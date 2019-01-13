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

def preprocess(data):
    stock_df = StockDataFrame.retype(data)
    stock_df['rsi_14']
    stock_df['adx']
    data=data[['close','rsi_14','adx']]

    #create direction column
    direction = data['close'].pct_change()
    bins=[-1,0,1]
    data['direction'] = pd.cut(direction,bins,labels=[-1,1]).to_frame()
    data['direction']=data['direction'].shift(-1)
    data['rsi_change'] = data['rsi_14'].pct_change()

    data=data.replace([np.inf], np.nan).replace([0],np.nan).dropna()

    #Normalize Data
    MinMax=preprocessing.MinMaxScaler()
    data[['rsi_14','rsi_change','adx']]=MinMax.fit_transform(data[['rsi_14','rsi_change','adx']])

    return data[['close','rsi_14','adx','direction','rsi_change']]


############################################################

def combination(lst):
    final=[]
    local=copy.deepcopy(lst)    #so original list is not disturbed
    return combinationHelper(local,final)

def combinationHelper(lst,final):
    if lst==[]:
        final.remove([])
        return final
    elif final==[]:
        final.append([])
    else:
        item=lst.pop()
        c=copy.deepcopy(final)
        for f in c:
            new_item=f
            new_item.append(item)
            final.append(new_item)
    return combinationHelper(lst,final)


#########################################################

def gridSearch(features,tests,paramGrid,data):
    dfdict={'test':[],'accuracy':[],'feature':[]}
    y=data['direction']

    # best model with validation
    print('...')
    for feature in combination(features):

        X=data[feature]
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
                    dfdict['feature'].append(feature)
                    dfdict['test'].append(copy.deepcopy(test))
                    #deepcopy prevents changing dataframe

    df=pd.DataFrame(dfdict)
    return df

def findBestModel(df,data):
    best_validation_score=df['accuracy'].max()
    best_index=df.index[df['accuracy']==best_validation_score].tolist()[0]

    ##########
    best_series=df.iloc[best_index]
    best_feature=best_series['feature']
    best_test=best_series['test']

    X=data[best_feature]
    y=data['direction']

    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.40,shuffle=False)
    XValidate,XTest,yValidate,yTest = model_selection.train_test_split(XTest,yTest,test_size=0.50,shuffle=False)

    # best model with testing set
    best_test.fit(XTrain,yTrain)
    yPredict=best_test.predict(XTest)
    best_test_score=metrics.accuracy_score(yTest,yPredict)

    print('best test : ' + str(best_test))
    print('best feature : ' + str(best_feature))
    print('validation score : ' + str(best_validation_score))
    print('test score : ' + str(best_test_score))

    ###############

    XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=0.20,shuffle=False)

    # best model with testing set
    best_test.fit(XTrain,yTrain)
    yPredict=best_test.predict(XTest)
    best_test_score2=metrics.accuracy_score(yTest,yPredict)

    print('test score : ' + str(best_test_score2))
    return best_test_score, best_test_score2


############################################################

def execution():
    featList=['rsi_14','rsi_change','adx']

    clfList=[neighbors.KNeighborsClassifier(),
            gaussian_process.GaussianProcessClassifier(),
            ensemble.RandomForestClassifier(),
            naive_bayes.GaussianNB(),
            linear_model.LogisticRegression()]

    gridList=[{'n_neighbors': [n for n in range(1,30,2)]}, 
                #'weights':['uniform','distance']},
            {'kernel':[None]},
            {'n_estimators': [n for n in range(1,25)]},
            {'priors':[None]},
            {'C': [n for n in np.arange(0.1,1.1,0.1)]}]

    stockList=['baba','wmt','ba','roku','vz','ibm','bidu']
        #'amzn','goog','orcl','fb','twtr',
        #'cmg','gis','k','khc','mcd','hsy','tsn']

    meanAcc=0
    meanAcc2=0

    divisor=len(stockList)
    for stock in stockList:
        print(stock)
        data=iex.stock_info(stock).create_ranged_dataset('5y')
        data=preprocess(data)
        print('shape: ' + str(data.shape))
        if data.shape[0]<1000:
            divisor-=1
            continue

        df=gridSearch(featList,clfList,gridList,data)
        acc=findBestModel(df,data)
        meanAcc+=acc[0]
        meanAcc2+=acc[1]

    print('---')
    print('average accuracy: %.8f' % (meanAcc/float(divisor)))
    print('average accuracy: %.8f' % (meanAcc2/float(divisor)))
execution()

# print confusion matrix
# fix gridSearch
# try more features

# test if combining data from all companies is good
# try probability threshold

# Stock-Market-Predictor
*The python packages sklearn and stockstats must be installed

This project is designed to help users make investment decisions by predicting whether a stock will increase or decrease using python's
machine learning package sklearn. 

How it works:
A sample of daily data from the past 5 years (not including this year) in 10 different stocks are used in training.
The direction of stock price (either increase or decrease from the previous day) are trained against different 
(1) Stock Momentum Indicators (RSI, ADX, ...).
(2) Multiple machine learning algorithms (KNN, Random Forest, ...) are tested, each with  
(3) different parameter values (For KNN, number of neighbors) to find the model with the highest accuracy. 
When the optimal model is found, the model is tested against this year's data to see how well the model performed.
(This is called a train-test split)

1) Run the program

# unit13-challenge


## Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the Crypto Fear and Greed Index (FNG) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. You have been asked to help build and evaluate deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.##

## In this assignment, you will use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price. ##
--------

## Technologies and Installation Guide
import numpy as np
import pandas as pd
import hvplot.pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow import random
from numpy.random import seed



## Observations
The models were tested on three windows of data: 1 day, 5 days, and 10 days.

![actual_vc_predictedCP1](../Images/actual_vc_predictedCP1.png)![actual_vc_predictedfng1](../Images/actual_vc_predictedfng1.png)
![actual_vc_predictedCP5](../Images/actual_vc_predictedCP5.png)![actual_vc_predictedfng5](../Images/actual_vc_predictedfng5.png)
![actual_vc_predictedCP10](../Images/actual_vc_predictedCP10.png)![actual_vc_predictedfng10](../Images/actual_vc_predictedfng10.png)



### Conclusions ###


Which model has a lower loss?
The model with the lower loss is the closing prices predictor

Which model tracks the actual values better over time?
The model with better results over time is the closing prices predictor


Which window size works best for the model?
The window size that works better is 1

## Contributors##

By: Roy Booker

---

## License ##

MIT

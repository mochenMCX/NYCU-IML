from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from loguru import logger

# read data
train_df = pd.read_csv('./train.csv') #n*5
train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()   #delete title, change to numpy
train_y = train_df["Performance Index"].to_numpy()  #n*1

model = LinearRegression()
model.fit(train_x, train_y)

# 打印权重和截距
print("Weights (β):", model.coef_)          # 打印贝塔权重
print("Intercept (b):", model.intercept_)    # 打印截距
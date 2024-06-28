import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def data_splitter(X,y,test_size=20):
    test_size /= 100
    return train_test_split(X, y, test_size=test_size, random_state=0)


def LR_model_maker():
    model = LinearRegression()
    return model

def LR_trainer(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def LR_tester(model, x_test, y_test):
    score = model.score(x_test, y_test)
    return score

# X = np.array([1,3,4,6,9,12,15,37,69,70]).reshape(-1, 1)
# y = [2,4,6,10,20,24,30,32,36,45]
#
#
# x_train, x_test, y_train, y_test = data_splitter(X,y,test_size=20)
# model = LR_model_maker()
# LR_trainer(model, x_train, y_train)
# print(LR_tester(model, x_test, y_test))

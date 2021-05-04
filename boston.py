from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston 
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn import preprocessing, cross_validation, svm


#TO RUN pip3 install -U scikit-learn

x_axis = [1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
tsla_prices= [597.95, 563.00, 673.58, 668.06, 699.60, 693.73, 707.94, 676.88, 701.81, 653.16, 654.87, 670.00, 662.16, 630.27, 640.39, 618.71, 611.29, 635.62, 667.93, 661.75, 691.05, 691.62, 670.97, 683.80, 677.02, 701.98, 762.32, 732.23, 738.85, 739.78,]

X = [[626.06, 597.95, 12,860.04, 12,920.15],
[600.55, 563.00, 12,904.26, 12,609.16],
[608.18, 673.58, 12,923.07, 13,073.82],
[700.30, 668.06, 13,234.73, 13,068.83],
[699.40, 699.60, 13,273.31, 13,398.67],
[670.00, 693.73, 13,222.81, 13,319.86],
[694.09, 707.94, 13,323.47, 13,459.71],
[703.35, 676.88, 13,523.17, 13,471.57],
[656.87, 701.81, 13,336.91, 13,525.20],
[684.29, 653.16, 13,349.20, 13,116.17],
[646.60, 654.87, 13,119.90, 13,215.24],
[684.59, 670.00, 13,278.78, 13,377.54],
[675.77, 662.16, 13,381.43, 13,227.70],
[667.91, 630.27, 13,289.24, 12,961.89],
[613.00, 640.39, 12,844.58, 12,977.68],
[641.87, 618.71, 12,996.03, 13,138.73],
[615.64, 611.29, 13,103.97, 13,059.65],
[601.75, 635.62, 13,008.80, 13,045.39],
[646.62, 667.93, 13,122.57, 13,246.87],
[688.37, 661.75, 13,414.32, 13,480.11],
[707.71, 691.05, 13,594.90, 13,705.59],
[690.30, 691.62, 13,681.67, 13,698.38],
[687.00, 670.97, 13,675.30, 13,688.84],
[677.38, 683.80, 13,796.89, 13,829.31],
[677.77, 677.02, 13,787.02, 13,900.19],
[685.70, 701.98, 13,854.44, 13,850.00],
[712.70, 762.32, 13,902.45, 13,996.10],
[770.70, 732.23, 14,004.08, 13,857.84],
[743.10, 738.85, 13,983.23, 14,038.76],
[728.65, 739.78, 14,059.11, 14,052.34]]

# X = [
# [626.06, 597.95],
# [600.55, 563.00],
# [608.18, 673.58],
# [700.30, 668.06],
# [699.40, 699.60],
# [670.00, 693.73],
# [694.09, 707.94],
# [703.35, 676.88],
# [656.87, 701.81],
# [684.29, 653.16],
# [646.60, 654.87],
# [684.59, 670.00],
# [675.77, 662.16],
# [667.91, 630.27],
# [613.00, 640.39],
# [641.87, 618.71],
# [615.64, 611.29],
# [601.75, 635.62],
# [646.62, 667.93],
# [688.37, 661.75],
# [707.71, 691.05],
# [690.30, 691.62],
# [687.00, 670.97],
# [677.38, 683.80],
# [677.77, 677.02],
# [685.70, 701.98],
# [712.70, 762.32],
# [770.70, 732.23],
# [743.10, 738.85],
# [728.65, 739.78]]

y = tsla_prices

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.01 )


# sgdr = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1, penalty='elasticnet')
# sgdr.fit(xtrain, ytrain)

# ypred = sgdr.predict(xtest)
# print(ypred)

# y_var = sgdr.score(xtrain, ytrain)
# r2 = sgdr.coef_
# intercept = sgdr.intercept_


###################
#SGD
# clf = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1, penalty='elasticnet')
# clf.fit(xtrain, ytrain)
# prediction = (clf.predict(xtrain))
# y_var = clf.score(xtrain, ytrain)
# r2 = clf.coef_
# intercept = clf.intercept_
# plt.scatter(y[:-1],prediction)
# plt.grid()
# plt.xlabel('Actual y')
# plt.ylabel('Predicted y')
# plt.title('Scatter plot from actual y and predicted y')
# plt.show()

##############################################
###LINEAR REG - I work
clf = LinearRegression()
clf.fit(xtrain, ytrain)
prediction = (clf.predict(xtrain))
y_var = clf.score(xtrain, ytrain)
r2 = clf.coef_
intercept = clf.intercept_
# print(y_var)
# print(r2)
# print(intercept)
plt.scatter(y[:-1],prediction)

a=list(y[:-1])
z=list(prediction)
plt.plot(np.unique(a), np.poly1d(np.polyfit(a, z, 1))(np.unique(a)))


plt.grid()
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('SGD REGRESSION TESLA')
plt.show()
##############################################


# plt.plot(x_axis, tsla_prices, label="original")
# plt.plot(len(ypred), ypred, label="predicted")
# plt.plot(x_axis[:-1], ytrain, label="predicted")
# plt.plot(x_axis[:-1], prediction, label="predicted")
# plt.title("TSLA PRICES")
# plt.xlabel('DAY')
# plt.ylabel('COST $')
# plt.legend(loc='best',fancybox=True, shadow=True)
# plt.grid(True)
# plt.show() 


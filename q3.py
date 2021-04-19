import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf

def stoc_gradient():
    # Create needed objects
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    var = tf.Variable(2.5)
    cost = lambda: 2 + var ** 2

    # Perform optimization
    for _ in range(100):
        sgd.minimize(cost, var_list=[var])

def plot_line_graph(data_set_name, y, x, y2=None):
    plt.plot(x,y,'b-', label = 'RAW')

    if y2 != None:
        plt.plot(y2, 'y--', label = 'Linear Reg.') #y is color, -- is line style

    plt.grid(True) #grid on
    plt.xlim(1, 30) #limits are [ymin, ymax]
    plt.xlabel('DAY')
    plt.ylabel('Price $')
    plt.title(data_set_name)
    plt.legend()
    return(plt.show())


# (A) plot original
orig_dates = [1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
tsla_prices= [597.95, 563.00, 673.58, 668.06, 699.60, 693.73, 707.94, 676.88, 701.81, 653.16, 654.87, 670.00, 662.16, 630.27, 640.39, 618.71, 611.29, 635.62, 667.93, 661.75, 691.05, 691.62, 670.97, 683.80, 677.02, 701.98, 762.32, 732.23, 738.85, 739.78,]
plot_line_graph('TSLA (3.5 - 4.16)', tsla_prices, orig_dates)

# (B)
tsla_linear_reg = []

sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
var = tf.Variable(2.5)
cost = lambda: 2 + var ** 2

# Perform optimization
for _ in range(100):
    sgd.minimize(cost, var_list=[var])


# (C) Plot raw data and linear
plot_line_graph('TSLA (3.5 - 4.16)', tsla_prices, orig_dates, tsla_linear_reg)
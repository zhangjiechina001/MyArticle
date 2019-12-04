import matplotlib.pyplot as plt
import numpy as np

# generate different normal distributions
x1 = np.random.normal(30, 3, 1000)
x2 = np.random.normal(20, 2, 1000)
x3 = np.random.normal(10, 3, 1000)

# plot them
plt.plot(x1, label='plot')
plt.plot(x2, label='2nd plot')
plt.plot(x3, label='last plot')

# generate a legend box
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
           ncol=3, mode="expand", borderaxespad=0.)

# annotate an important value
plt.annotate("Important value", (55, 20), xycoords='data',
             xytext=(5, 38),
             arrowprops=dict(arrowstyle='->'))
plt.show()

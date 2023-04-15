# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# x = np.arange( 1 , 11 , 2 )
# y = np.random.rand( 5 )
# y*= 5
# plt.plot(x, y, linestyle="--", marker="*", color="b")

# plt.title("Taras the Almighty")
# plt.xlabel("Taras")
# plt.ylabel("Panis")
# plt.grid(True)

# y1 = np.random.rand(11)
# y2 = np.random.rand(11)

# plt.subplot(1, 2, 1)
# plt.plot(y1, y2)

# plt.subplot(1, 2, 2)
# plt.plot(y1, y2)

# fig = plt.figure()
# sp1 = fig.add_subplot(1, 2, 1)
# sp1.plot(y1)
# sp2 = fig.add_subplot(1, 2, 2)
# sp2.plot(y2)

# # sample data 
# x = np.array([ 20 , 30 , 40 , 50 , 60 ])
# y = np.array([ 30 , 38 , 41 , 49 , 62 ])

# # get a tuple containing a and b in the approximate line formula 
# p = np.polyfit(x, y, 1 )

# # create an object for the expression of the primary function
# f = np.poly1d(p)

# # Draw a scatterplot and approximate line
# plt. scatter(x, y)
# plt.plot(x, f(x), marker="*")

# #plot 1:
# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(1, 2, 1)
# plt.plot(x,y)
# plt.title("SALES")

# #plot 2:
# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(1, 2, 2)
# plt.plot(x,y)
# plt.title("INCOME")

# plt.suptitle("MY SHOP")

# Pie chart

y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.2, 0, 0, 0]

plt.pie(y, labels = mylabels, explode = myexplode)

# Show chart
plt.show()

#!/usr/bin/env python

"""
Plotting live Twitter data in real-time
"""

__author__ = "Manan Kalra"
__email__ = "manankalr29@gmail.com"


from matplotlib import pyplot, animation, style


style.use("ggplot")
fig = pyplot.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open("twitter_out.txt", "r").read()
    lines = graph_data.split("\n")
    x_arr, y_arr = [], []
    x, y = 0, 0
    for l in lines:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 0.5
        x_arr.append(x)
        y_arr.append(y)
    ax1.clear()
    ax1.plot(x_arr, y_arr)

ani = animation.FuncAnimation(fig, animate, interval=1000)
pyplot.show()

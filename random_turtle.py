from turtle import *
import random

stop_flag = False

def clicked(x, y):
    global stop_flag
    stop_flag = True

onscreenclick(clicked)

speed(0)

while not stop_flag:
    left(random.randint(-90, 90))
    forward(10)
    if position()[0]**2 + position()[1]**2 > 200 **2:
        forward(-10)

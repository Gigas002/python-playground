# %% Test cell 1

# print("test message 1")

# %%

# test comment

# print("test message 2")

const_num = 2

def test_func(input):
    val1 = input
    val2 = const_num/val1
    return (val1 + val2)/2

age = {"山田":18, "田中":19}
age["佐藤"] = 20
# print(age)

# for/foreach loop
var = 0

# for i in range(10):
#     var += i
#     print(var, i)

# a = [i*i for i in range(5)]

# print(a)

# thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
# thislist.sort()
# print(thislist)

class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  def __init__(self, fname, lname, year):
    Person.__init__(self, fname, lname)
    self.graduationyear = year

# x = Student("Mike", "Olsen", 2019)
# x.printname()
# print(x.graduationyear)

import mymodule as mx
# x = mx.Taras("Panis")
# x.saymyname()

import platform
x = dir(mx)
# print(x)

from mymodule import person1
# print (person1["age"])

try:
    print(taras_panis)
    raise Exception("Test exception")
except:
    print("An exception occurred")

username = input("Enter username:")
print("Username is: " + username)

import os
if os.path.exists("demofile.txt"):
    os.remove("demofile.txt")
else:
    print("The file does not exist")

def testfunc():
    """
    Test decription
    """
    pass

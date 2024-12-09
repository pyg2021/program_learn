import numpy as np
y=0
all=15
# while y<40:
#     y+=1
#     all=1.05*all
#     print(y,all,all*0.05)
m=30
a=1.05
while y<10:
    all=a*all
    all+=m
    y+=1
    print(y,all,all*0.05)
while y<30:
    y+=1
    all=a*all
    print(y,all,all*0.05)
import numpy as np
x=np.ones((200,2,100,100))
print(x.mean(axis=(0,1)).shape)
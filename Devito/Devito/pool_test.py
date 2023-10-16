from multiprocessing import Pool
import multiprocessing as mp
import time
from tqdm import tqdm
start=time.time()
def myf(x):
    time.sleep(1)
    print(x)
    
    return x * x

if __name__ == '__main__':
    print('-- start run ---')
    value_x = range(100)
    # for i in range(100):
    #     myf(i)



    P = Pool(processes=30)
    # value_y = P.map(func=myf, iterable=value_x)
    for i in range(100):
        value_y=P.apply_async(func=myf, args=(i,))
    P.close()
    P.join()
    print(value_y.get())
    print(time.time()-start,"s")
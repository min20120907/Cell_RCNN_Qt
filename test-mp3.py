# load libraries
from multiprocessing import freeze_support
from multiprocessing import Pool
from functools import partial
import threading
import math
import pandas as pd
import numpy as np
from random import randint
from time import sleep

# define a test function that determines whether or not an integer is a prime number
def is_prime(n):
    if (n < 2) or (n % 2 == 0 and n > 2):
        return (n, False)
    elif n == 2:
        return (n, True)
    elif n == 3:
        return (n, True)
    else:
        for i in range(3, math.ceil(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return (n, False)
        return (n, True)

# set the maximum range to detect prime numbers in (0 to num_max-1)
num_max = 1000000
# looping
np.array([is_prime(x) for x in list(range(num_max))])
#mapping
np.array(list(map(is_prime, list(range(num_max)))))
# vectorizing
np.vectorize(is_prime)(list(range(num_max)))

# mutithreading version
def func_thread(n, out):
    out.append(is_prime(n))
x_ls =list(range(num_max))
thread_list = []
results = []
for x in x_ls:
    thread = threading.Thread(target=func_thread, args=(x, results))
    thread_list.append(thread)
for thread in thread_list:
    thread.start()
for thread in thread_list:
    thread.join()

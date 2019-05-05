import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

import numpy as np
import matplotlib.pyplot as plt


with open('makespan.txt','r') as file_to_read:
    makespan = []
    k=0
    while True:
        lines = file_to_read.readline()
        if not lines:
            break
            pass
        for i in lines.split():
            ms = float(i)
            if ms < 1000:
                makespan.append(ms)
                k=k+1
            
    
    plt.plot(range(k),makespan)
    
    plt.savefig("makespan.png")
    plt.show()


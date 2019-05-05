# coding=utf-8
import os
r = 30  # 容量
e = 100  # 模拟次数
for n in range(10, 11, 10):
    # if n > 50:
    #     s = 15000
    for p in range(3, 4):
        # for cp in range(-50, -4, 5):
            # for i in range(100):
        print("p=", p)
        command = "./daggen --dot -jump 2 " + " -n " + str(n) + " -p " + str(p) + " -r" + str(r) + " -e" + str(e)
        os.system(command)




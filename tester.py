# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:26:54 2017

@author: Jadon
"""

import sys
import random

def generate_samples():
    num_g1 = 100
    m_g1 =0
    std_g1 = 1
    num_g2 = 10
    m_g2=5
    std_g2 = 1
    for i in range(num_g1):
        x = random.gauss(m_g1,std_g1)
        print ("x:{0} from{1}".format(str(x), str(1)))
    for i in range(num_g2):
        x = random.gauss(m_g2,std_g2)
        print ("x:{0} from{1}".format(str(x), str(2)))
def main(op):
    #op = sys.argv[1]
    if op == "gen":
        generate_samples()

if __name__ =="__main__":
    main("gen")
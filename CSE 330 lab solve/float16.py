# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:58:17 2019

@author: PRANTO DEV
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math
import itertools
import matplotlib.pyplot as plt



class Float16():
   

    def __init__(self, bitstring):
        '''Constructor
        takes a 16-bit string of 0's and 1's as input and stores the sub-strings
        accordingly.
        Usage: Float16('0001111000011110')
        '''

        # Make sure the input consists of exactly 16-bits.
        assert(len(bitstring)==16)

        # Assign the sign bit
        # self.sign = bitstring[?]
        self.sign = bitstring[0]

        # Assign the exponent part
        # self.exponent = bitstring[?]
        self.exponent = bitstring[1:5]

        # Assign the mantissa
        # self.mantissa = bitstring[?]
        self.mantissa = bitstring[5:]


        self.val = self.calculate_value()
        
    def __str__(self):
        return f'Sign bit value: {self.sign}\n' + \
            f'Exponent value: {self.exponent}\n' + \
            f'Mantissa value: {self.mantissa}\n' + \
            f'Floating value: {self.val}\n'
    
    def tobitstring(self):
        return self.sign + self.exponent + self.mantissa
    
    def toformattedstring(self):
        return ' '.join([self.sign, self.exponent, self.mantissa])
    
    def calculate_value(self):
        '''Calculate the value of the number from bits'''

        # Initialize with zero
        val = 0.0

        # Handle the case of inf and nan
        # use math.inf to represent infinity
        # use math.nan to represent not-a-number

        if self.exponent == "1111":
            if self.mantissa=="00000000000":
                
                if self.sign == "1" :
                    val = -math.inf
                else :
                    val = math.inf
                return val
            else:
                return math.nan                
            
  

        # Handle the case of subnormals==> 

        elif self.exponent == "0000":
            val=0.0
            x=0
            y=1
            while x<11:
                temp=self.mantissa[x]
                val+=(int(temp)*math.pow(2,-y))
                x+=1
                y+=1
                
            
            val*=math.pow(2,-6)
            
            
  
       
        #    you can use a for loop to calculate the value.
        #    it should look like this.
        #    for exp, bit in enumerate(self.mantissa):
        #        val += ??
        #    val *= ??

        # Handle the case of normals

        else:
            val=1.0
            a=[8,4,2,1]
            x=0
            
            e=0
            temp=0
            while x<4:
                temp=self.exponent[x]
                e+=int(temp)*a[x]
                x+=1
            x=0
            
            y=1
            while x<11:
                temp=self.mantissa[x]
                val+=(int(temp)*math.pow(2,-y))
                x+=1
                y+=1
            val*=math.pow(2,e-7)
           
        #    val = 1.0
        #    
        #    use a loop like this to calculate mantissa value
        #    for exp, bit in enumerate(self.mantissa):
        #        val += ??
        #    
        #    use a loop like this to calculate exponent value
        #    for e, bit in enumerate(reversed(self.exponent)):
        #        exp += ??
        #    
        #    calculate final value
        #    val *= ??

        # Handle the sign bit

        if self.sign == "0":
            val*=math.pow(-1,0)
        else:
            val*=math.pow(-1,1)
            
     

        return val

        


def test1a():
    count = 0
    data = [ '0011100000000010', '0100000000000000', '1100000000000000', '0100010000000000',
             '1100010000000000', '0100100000000000', '1100100000000000', '0100101000000000',
             '1100101000000000', '0100110000000000', '1100110000000000', '0101101110000000',
             '0010010000000000', '0000000000000001', '0000011111111111', '0000100000000000',
             '0111011111111111', '0000000000000000', '1000000000000000', '0111100000000000',
             '1111100000000000', '0111100000000001', '0111110000000001', '0111111111111111',
             '0010101010101011', '0100010010010001', '0011100000000000', '0011100000000001']
    result = ['(1025, 1024)', '(2, 1)', '(-2, 1)', '(3, 1)', '(-3, 1)', '(4, 1)', '(-4, 1)',
               '(5, 1)', '(-5, 1)', '(6, 1)', '(-6, 1)', '(23, 1)', '(3, 16)', '(1, 131072)',
               '(2047, 131072)', '(1, 64)', '(4095, 16)', '(0, 1)', '(0, 1)', 'inf', '-inf',
               'nan', 'nan', 'nan', '(2731, 8192)', '(3217, 1024)', '(1, 1)', '(2049, 2048)']
    
    test = [Float16(x).val for x in data]
    for index in range(len(test)):
        d = test[index]
        try:
            test[index] = str(d.as_integer_ratio())
        except Exception:
            test[index] = str(d)
        if test[index] == result[index]:
            count += 1
        else:
            print(data[index], result[index], test[index])
    print(count, 'out of 28')

def histogram():
    combinations = itertools.product('01', repeat=16)
    bitstrings = [''.join(x) for x in combinations]
    numbers = list(map(Float16, bitstrings))
    values = [x.val for x in numbers]
    positive = values[0:30720]
    negative = values[32768:63488]
    plt.hist(positive, 64)
    plt.hist(negative, 64)
test1a()
histogram()

        
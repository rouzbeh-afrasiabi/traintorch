import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from traintorch import *
import numpy as np
import pandas as pd

class TestClass:
    def test_metric(self):
        test=metric('test',w_size=10,average=False,xaxis_int=True,n_ticks=(5, 5))
        assert test.name=='test'
        assert test.w_size==11
        assert test.average==False
        assert test.xaxis_int==True
        assert test.n_ticks==(5,5)

        for i in range(0,100):
            pass
            test.update(x=2*i,t=i,f=3*i)
        assert all(np.hstack(test.means)==[5,15,16,48,27,81,38,114,49,147,60,180,71,213,82,246,93,279])
        assert test.counter==100
        assert test.keys==['x', 't', 'f']
        assert test.updated==True
        assert all(np.hstack(test.window().values==[[ 89, 267],[ 90, 270],[ 91, 273],[ 92, 276],[ 93, 279],
                               [ 94, 282],[ 95, 285],[ 96, 288],[ 97, 291],[ 98, 294],[ 99, 297]]))
        assert test.x==[198]
        
    def test_two(self):
        x = 1
        assert x==1

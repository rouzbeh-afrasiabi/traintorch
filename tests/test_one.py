import pytest
from traintorch import *

class TestClass:
    def test_one(self):
        test=metric('test',w_size=10,average=False,xaxis_int=True,n_ticks=(5, 5))
        assert test.name=='test'
        assert test.w_size==11
        assert test.average==False
        assert test.xaxis_int==True
        assert test.n_ticks==(5,5)

    def test_two(self):
        x = 1
        assert x==1

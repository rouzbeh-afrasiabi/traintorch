# Traintorch (alpha)


Package for live visualization of model validation metrics during training of a machine learning model in jupyter notebooks.
 

## Requirements:

```
pandas==0.25.1
matplotlib==3.1.1
ipython==7.8.0
numpy==1.17.2
pycm==2.2
```
 ## Installation:
 ```
 pip install traintorch
  ```
or

 ```
 pip install git+https://github.com/rouzbeh-afrasiabi/traintorch.git
 ```

## Example 
```python
from traintorch import *

first=metric('first',w_size=100)
second=metric('second',w_size=100)
third=metric('third',w_size=5)

tracker=traintorch(n_custom_plots=3,main_grid_hspace=.1,
              figsize=(15,15))

for i in range(0,5000,1):

    first.update(x=i,b=i*100,g=i**2)
    second.update(train=1/(i+1),test=1/(i**2+1))
    
    if((i+1)%20==0 and i>0):
        third.update(result=np.sin(i%3))
        tracker.create(custom_metrics=[first,second,third])
```
 <p align='center'>
 <img src='./images/dash_a.png'></img>
 
 </p>

# Traintorch v.1.0.2-alpha
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d4b74c08973343128d17532b4b84e154)](https://www.codacy.com/manual/rouzbeh-afrasiabi/traintorch?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=rouzbeh-afrasiabi/traintorch&amp;utm_campaign=Badge_Grade)


<p align="justify">
Package for live visualization of model validation metrics during training of a machine learning model in jupyter notebooks. Traintorch utilizes a sliding window mechanism to reduce memory usage.
</p> 

## Requirements:

```
pandas==0.25.1
matplotlib==3.1.1
ipython==7.8.0
numpy==1.17.2
pycm==2.2
```
 ## Installation:
 
 ### Latest release:
 ```
 pip install traintorch
  ```
  
### Latest Version

 ```
 pip install git+https://github.com/rouzbeh-afrasiabi/traintorch.git
 ```
### List of Metrics available through pycm

'<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>pycm_metrics_0</th>\n      <th>pycm_metrics_1</th>\n      <th>pycm_metrics_2</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>ACC Macro</td>\n      <td>Lambda A</td>\n      <td>Standard Error</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>AUNP</td>\n      <td>Lambda B</td>\n      <td>TPR Macro</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>AUNU</td>\n      <td>Mutual Information</td>\n      <td>TPR Micro</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>Bennett S</td>\n      <td>NIR</td>\n      <td>Zero-one Loss</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>CBA</td>\n      <td>Overall ACC</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>Chi-Squared</td>\n      <td>Overall CEN</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>6</th>\n      <td>Chi-Squared DF</td>\n      <td>Overall MCC</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>7</th>\n      <td>Conditional Entropy</td>\n      <td>Overall MCEN</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>8</th>\n      <td>Cramer V</td>\n      <td>Overall RACC</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>9</th>\n      <td>Cross Entropy</td>\n      <td>Overall RACCU</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>10</th>\n      <td>F1 Macro</td>\n      <td>P-Value</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>11</th>\n      <td>F1 Micro</td>\n      <td>PPV Macro</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>12</th>\n      <td>Gwet AC1</td>\n      <td>PPV Micro</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>13</th>\n      <td>Hamming Loss</td>\n      <td>Pearson C</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>14</th>\n      <td>Joint Entropy</td>\n      <td>Phi-Squared</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>15</th>\n      <td>KL Divergence</td>\n      <td>RCI</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>16</th>\n      <td>Kappa</td>\n      <td>RR</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>17</th>\n      <td>Kappa No Prevalence</td>\n      <td>Reference Entropy</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>18</th>\n      <td>Kappa Standard Error</td>\n      <td>Response Entropy</td>\n      <td></td>\n    </tr>\n    <tr>\n      <th>19</th>\n      <td>Kappa Unbiased</td>\n      <td>Scott PI</td>\n      <td></td>\n    </tr>\n  </tbody>\n</table>'

## Example 

### Simple Usage
```python
from traintorch import *

#custom metrics
first=metric('Loss',w_size=10,average=False)
second=metric('Accuracy',w_size=10,average=False)


#create an instance of traintorch
tracker=traintorch(n_custom_plots=2,main_grid_hspace=.1, figsize=(15,10),show_table=True)
#combine all metrics together
tracker.append([first,second])


range_max=1000
for i in range(0,range_max,1):
    
    first.update(train_loss=1/(i+1),test_loss=1/(i**2+1))
    second.update(y=i/(i*2+1))
    tracker.plot()
```
 <p align='center'>
 <img src='./images/dash_a.png'></img>
 
 </p>



### Using pycm metrics and doing comparison


```python
from traintorch import *


#custom metric
first=metric('Loss',w_size=10,average=False)

#pycm metrics
overall_selected=['ACC Macro']
cm_metrics_a=pycmMetrics(overall_selected,name='train',w_size=10)
cm_metrics_b=pycmMetrics(overall_selected,name='test',w_size=10)

#compare two metrics of the same kind
compare_a=collate(cm_metrics_a,cm_metrics_b,'ACC Macro')

#create an instance of traintorch
tracker=traintorch(n_custom_plots=1,main_grid_hspace=.1,figsize=(15,15),show_table=True)

#combine all metrics together
tracker.append([first,cm_metrics_a,cm_metrics_b,compare_a])


range_max=1000
for i in range(0,range_max,1):
    
    actual_a=np.random.choice([0, 1], size=(20,), p=[1./3, 2./3])
    predicted_a=np.random.choice([0, 1], size=(20,),p=[1-(i/range_max), i/range_max])
    actual_b=np.random.choice([0, 1], size=(20,), p=[1./3, 2./3])
    predicted_b=np.random.choice([0, 1], size=(20,),p=[1-(i/range_max), i/range_max])
    cm_metrics_a.update(actual_a,predicted_a)
    cm_metrics_b.update(actual_b,predicted_b)
    first.update(train=1/(i+1),test=1/(i**2+1))
    compare_a.update()
    tracker.plot()

```
 <p align='center'>
 <img src='./images/dash.png'></img>
 
 </p>

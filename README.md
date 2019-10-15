-------------
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d4b74c08973343128d17532b4b84e154)](https://www.codacy.com/manual/rouzbeh-afrasiabi/traintorch?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=rouzbeh-afrasiabi/traintorch&amp;utm_campaign=Badge_Grade)

[![HitCount](http://hits.dwyl.io/rouzbeh-afrasiabi/traintorch.svg)](http://hits.dwyl.io/rouzbeh-afrasiabi/traintorch)

-------------

# TrainTorch v.1.0.2-alpha


<p align="justify">
Package for live visualization of model validation metrics during training of a machine learning model in jupyter notebooks. Traintorch utilizes a sliding window mechanism to reduce memory usage.
</p> 

## Requirements

```
pandas==0.25.1
matplotlib==3.1.1
pycm==2.2
```
 ## Installation
 
 ### Latest release
 ```
 pip install traintorch
  ```
  
### Latest Version

 ```
 pip install git+https://github.com/rouzbeh-afrasiabi/traintorch.git
 ```
 
-------------

## Examples

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

### Using pycm metrics and performing comparison

<br>

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

-------------
<br>

## Metrics available through pycm
<br>

|    | pycm_metrics_0      | pycm_metrics_1       | pycm_metrics_2     | pycm_metrics_3    | pycm_metrics_4   |
|:---|:--------------------|:---------------------|:-------------------|:------------------|:-----------------|
| 0  | ACC Macro           | F1 Macro             | Lambda A           | P-Value           | Standard Error   |
| 1  | AUNP                | F1 Micro             | Lambda B           | PPV Macro         | TPR Macro        |
| 2  | AUNU                | Gwet AC1             | Mutual Information | PPV Micro         | TPR Micro        |
| 3  | Bennett S           | Hamming Loss         | NIR                | Pearson C         | Zero-one Loss    |
| 4  | CBA                 | Joint Entropy        | Overall ACC        | Phi-Squared       |                  |
| 5  | Chi-Squared         | KL Divergence        | Overall CEN        | RCI               |                  |
| 6  | Chi-Squared DF      | Kappa                | Overall MCC        | RR                |                  |
| 7  | Conditional Entropy | Kappa No Prevalence  | Overall MCEN       | Reference Entropy |                  |
| 8  | Cramer V            | Kappa Standard Error | Overall RACC       | Response Entropy  |                  |
| 9  | Cross Entropy       | Kappa Unbiased       | Overall RACCU      | Scott PI          |                  |

<br>
<p align="justify">
Following class metrics are also available through pycm but their use is currently not recommended.
</p> 
<br>

|    | pycm_metrics_0   | pycm_metrics_1   | pycm_metrics_2   | pycm_metrics_3   | pycm_metrics_4   |
|:---|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| 0  | TPR              | MCC              | POP              | ERR              | GM               |
| 1  | TNR              | BM               | P                | RACCU            | Q                |
| 2  | PPV              | MK               | N                | J                | AGM              |
| 3  | NPV              | PLR              | TOP              | IS               | NLRI             |
| 4  | FNR              | NLR              | TON              | CEN              | MCCI             |
| 5  | FPR              | DOR              | PRE              | MCEN             |                  |
| 6  | FDR              | TP               | G                | AUC              |                  |
| 7  | FOR              | TN               | RACC             | sInd             |                  |
| 8  | ACC              | FP               | F0.5             | dInd             |                  |
| 9  | F1               | FN               | F2               | DP               |                  |

For more information about these metrics please see: <a href="https://github.com/sepandhaghighi/pycm">pycm</a>

from pycm import *
import gc
import pandas as pd
import numpy as np
from traintorch.TrainTorch import metric

class pycmMetrics():
    def __init__(self,overall_metrics,class_metrics,w_size=10):
        
        self._overall_metrics=['ACC Macro', 'AUNP', 'AUNU', 'Bennett S', 'CBA', 'Chi-Squared',
                        'Chi-Squared DF', 'Conditional Entropy', 'Cramer V',
                        'Cross Entropy', 'F1 Macro', 'F1 Micro', 'Gwet AC1',
                        'Hamming Loss', 'Joint Entropy', 'KL Divergence', 'Kappa',
                        'Kappa No Prevalence', 'Kappa Standard Error', 'Kappa Unbiased',
                        'Lambda A', 'Lambda B', 'Mutual Information', 'NIR', 'Overall ACC',
                        'Overall CEN', 'Overall MCC', 'Overall MCEN', 'Overall RACC',
                        'Overall RACCU', 'P-Value', 'PPV Macro', 'PPV Micro', 'Pearson C',
                        'Phi-Squared', 'RCI', 'RR', 'Reference Entropy',
                        'Response Entropy', 'Scott PI', 'Standard Error', 'TPR Macro',
                        'TPR Micro', 'Zero-one Loss']

        self._class_metrics=['TPR', 'TNR', 'PPV', 'NPV', 'FNR', 'FPR', 'FDR', 'FOR',
                        'ACC', 'F1', 'MCC', 'BM', 'MK', 'PLR', 'NLR', 'DOR', 'TP', 'TN', 'FP',
                        'FN', 'POP', 'P', 'N', 'TOP', 'TON', 'PRE', 'G', 'RACC', 'F0.5', 'F2',
                        'ERR', 'RACCU', 'J', 'IS', 'CEN', 'MCEN', 'AUC', 'sInd', 'dInd', 'DP',
                        'Y', 'PLRI', 'DPI', 'AUCI', 'GI', 'LS', 'AM', 'BCD', 'OP', 'IBA', 'GM',
                        'Q', 'AGM', 'NLRI', 'MCCI']
        
        self._all_metrics=self._overall_metrics+self._class_metrics
        self.cm_dict_overall={}
        self.cm_dict_class={}
        self.cm_df_overall=pd.DataFrame()
        self.cm_df_class=pd.DataFrame()
        self.w_size=w_size
        self.metrics={}
        self.metrics_oa={}
        self.metrics_cls={}
        self.overall_metrics=[]
        self.class_metrics=[]      
        self.overall_metrics=overall_metrics
        self.class_metrics=class_metrics 
        if(self.overall_metrics and self.class_metrics):
            for i,key in enumerate(self.overall_metrics):
                if(key in self._overall_metrics):
                    _key=str(key).replace(' ','_')
                    self.metrics_oa[_key]=metric(name=_key,w_size=self.w_size)
            for i,key in enumerate(self.class_metrics):
                if(key in self._class_metrics):
                    _key=str(key).replace(' ','_')
                    self.metrics_cls[_key]=metric(name=_key,w_size=self.w_size) 
        else:
            raise Exception("Please provide metric names.")
        gc.collect()
    def _in_list(self,target,main):
        return set(target)<set(main)
    def _to_list(self,target):
        if(isinstance(target, list)):
            pass
        else:
            target=[target]
        return target

    def _reform(self,target):
        return {(outerKey, innerKey): values for outerKey, innerDict in target.items() for innerKey, values in innerDict.items()}
    def _to_dict(self,_cm):
        _main={}
        _class={}
        for key,value in _cm.__dict__['overall_stat'].items():
            if(key in self.overall_metrics):
                if((not isinstance(value, tuple)) and (not isinstance(value, str))):
                    _main[str(key).replace(' ','_')]=value

        for key,value in _cm.__dict__['class_stat'].items():
            if(key in self.class_metrics):
                _class[str(key).replace(' ','_')]=value

        self.cm_dict_overall=_main
        self.cm_dict_class=_class
        gc.collect()
        
    def _to_df(self,_cm):
        
        _main,_class=self._to_dict(_cm)
        self.cm_dict_overall=_main
        self.cm_dict_class=_class
        
        self.cm_df_overall=pd.DataFrame(_main,index=[0])
        self.cm_df_class=pd.DataFrame(_class,index=[0])
        gc.collect()
        
    def update(self,actual,predicted):
        _cm=ConfusionMatrix(actual,predicted)
        self._to_dict(_cm)
        if(self.metrics_oa and self.metrics_cls):
            for k,v in self.cm_dict_overall.items():
                self.metrics_oa[k].update(**{k:v})

            for k,v in self.cm_dict_class.items():
                if(isinstance(v,dict)):
                    self.metrics_cls[k].update(**{'class_'+str(k_1):v_1 for k_1,v_1 in v.items()})        
                else:
                    self.metrics_cls[k].update(**{k:v})
            self.metrics=list({**self.metrics_oa,**self.metrics_cls}.values())
        else:
            raise Exception ("Metrics have not been declared.")
        gc.collect()
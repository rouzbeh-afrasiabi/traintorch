"""
Copyright (c) 2019 Rouzbeh Afrasiabi

https://github.com/rouzbeh-afrasiabi/

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""






from traintorch.utils import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import math
import sys
import os
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
import gc
from pycm import *


from IPython import get_ipython
import cv2
import re
import uuid
from datetime import datetime
import imageio

_ipy = get_ipython()
if _ipy is not None:
    get_ipython().run_line_magic('matplotlib','inline')

warnings.filterwarnings("ignore")


class traintorch:
    def __init__(self,name='',figsize=(15,20),show_table=True,n_custom_plots=2,
                 top_rows=1,top_cols=2,plot_width=4,plot_height=4,nrows=2,ncols=1,
                main_grid_hspace=0.5,main_grid_wspace=0.5,window=100,
                 garbage_collection=True,save_plots=False,save_snapshot=True,log_filename='main',
                 model_config={},custom_window=[]):
        if(name):
            self.name=name
        else:
            raise Exception ('please provide a name for traintorch instance.')     
        self.show_table=show_table
        self.top_rows=top_rows
        self.top_cols=top_cols
        self.plot_width=plot_width
        self.plot_height=plot_height
        self.nrows=nrows
        self.ncols=ncols
        self.top_axes=[]
        self.main_grid_hspace=main_grid_hspace
        self.main_grid_wspace=main_grid_wspace
        self.n_splits=22
        self.counter=0
        self.window=window
        self.custom_metrics=[]
        self.custom_tbl_metrics=[]
        self.main_results=pd.DataFrame()
        self.custom_data=None
        self.total_plots=0
        self.n_custom_plots=n_custom_plots
        self.figsize=figsize
        self.garbage_collection=garbage_collection
        self.save_plots=save_plots
        self.save_snapshot=save_snapshot
        #this needs to be fixed
        if(custom_window==[] and n_custom_plots>0):
            self.custom_window=[self.window]*n_custom_plots
        else:
            if(isinstance(custom_window, list)):
                self.custom_window=custom_window
            else:
                self.custom_window=[custom_window]

        self._avg_axes=[]
        self.save_folder=os.path.join(cwd,'save')
        self.project_folder=os.path.join(self.save_folder,self.name)
        self.run_folder=os.path.join(self.project_folder,'runs')
        self.timestamp=datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.uid=uuid.uuid4().hex

        self.main_folder=os.path.join(self.run_folder,self.uid)
        self.timestamp_folder=os.path.join(self.main_folder,self.timestamp)
        self.image_folder=os.path.join(self.timestamp_folder,'images')
        self.video_folder=os.path.join(self.timestamp_folder,'videos')
        self.checkpoint_folder=os.path.join(self.timestamp_folder,'checkpoints')
        self.data_folder=os.path.join(self.timestamp_folder,'data')
        self.snapshot_folder=os.path.join(self.timestamp_folder,'snapshot')

        self.model_config=model_config
        create_folders([self.save_folder,self.project_folder,self.run_folder,
                        self.main_folder,self.timestamp_folder,self.image_folder,
                        self.video_folder,self.data_folder,
                        self.checkpoint_folder,
                       ])

        #write to logfile line by line        
        self.log_filename=log_filename
        _log={'name':self.name,'uid':self.uid,'timestamp':self.timestamp,'model_config': self.model_config}
        log__(self.project_folder,_log,self.log_filename,custom_name=True) 
        
        #create a snapshot
        if(self.save_snapshot):
            snap__(self.snapshot_folder)
        
    def to_gif(self,name='',frame_rate=5):
        if(name):
            video_loc =os.path.join(self.video_folder,name+'.gif')
        else:
            video_loc =os.path.join(self.video_folder,self.uid+'.gif') 
        images = [img for img in os.listdir(self.image_folder) if img.endswith(".png")]
        images_lit=[(int(re.findall(r'\d+', image)[0]),image) for image in images]
        images_lit_sorted=sorted(images_lit, key=lambda image: image[0])
        images_list_sorted=[item[1] for item in images_lit_sorted]
        frame = imageio.imread(os.path.join(self.image_folder, images_list_sorted[0]))
        height, width, layers = frame.shape
        ims=[]
        if(images_list_sorted):
            for image in images_list_sorted:
                im=imageio.imread(os.path.join(self.image_folder, image))
                cv2.resize(im,(width,height))
                ims.append(im)
                
            imageio.mimsave(video_loc, ims,fps=frame_rate)
        else:
            raise Exception('No images found to convert to video.')
         
    def to_video(self,name='',frame_rate=5):
        if(name):
            video_loc =os.path.join(self.video_folder,name+'.avi')
        else:
            video_loc =os.path.join(self.video_folder,self.uid+'.avi') 
        images = [img for img in os.listdir(self.image_folder) if img.endswith(".png")]
        images_lit=[(int(re.findall(r'\d+', image)[0]),image) for image in images]
        images_lit_sorted=sorted(images_lit, key=lambda image: image[0])
        images_list_sorted=[item[1] for item in images_lit_sorted]
        if(images_list_sorted):
            frame = cv2.imread(os.path.join(self.image_folder, images_list_sorted[0]))
            height, width, layers = frame.shape
            #you can set fps here
            video = cv2.VideoWriter(video_loc, 0, frame_rate, (width,height),True)
            for image in images_list_sorted:
                im=cv2.imread(os.path.join(self.image_folder, image),cv2.IMREAD_COLOR)
                video.write(cv2.resize(im,(width,height)))
                
            cv2.destroyAllWindows()
            video.release()
        else:
            raise Exception('No images found to convert to video.')
    def log(self,**kwargs):
        log_filename='user'
        log__(self.data_folder,kwargs,log_filename,custom_name=True) 
        
    def append(self,plot_targets,table_targets=None):
        if(plot_targets):
            temp=[]
            for item in plot_targets:
                if(isinstance(item,(metric,collate))):
                    temp.append(item)
                elif(isinstance(item,pycmMetrics)):
                    temp+=item.metrics
                item.data_folder=self.data_folder
            self.custom_metrics=temp
            
        if(table_targets):
            temp=[]
            for item in table_targets:
                if(isinstance(item,metricsTable)):
                    temp.append(item)
            self.custom_tbl_metrics=temp

        class plot:
            def __init__(self,parent,):
                self.parent=parent
                self.main_grid_hspace=self.parent.main_grid_hspace
                self.main_grid_wspace=self.parent.main_grid_wspace
    
            def chunks_df(self,l, n,method='r'):
                if(method=='c'):
                    for i in range(0, l.shape[1], n):
                        yield l.iloc[:,i:i + n]
                elif(method=='r'):
                    for i in range(0, l.shape[0], n):
                        yield l.iloc[i:i + n,:]
            def tail(self,):
                main_results=pd.DataFrame()
                for _metric in self.parent.custom_metrics:
                    if(not isinstance(_metric,collate)):
                        temp=_metric.window().iloc[-1:,:]
                        temp.reset_index(drop=True, inplace=True)
                        main_results=pd.concat([main_results, temp], axis=1,sort=False)
                for _metric in self.parent.custom_tbl_metrics:
                    temp=_metric.window()
                    temp.reset_index(drop=True, inplace=True)
                    main_results=pd.concat([main_results, temp], axis=1,sort=False)
                self.parent.main_results=main_results
            
            def create(self,):
                #auto-update collate object
                for item in self.parent.custom_metrics:
                    if(isinstance(item,collate)):
                        item.update() 
                    
                if(any([item.updated for item in self.parent.custom_metrics])):
                    if(self.parent.garbage_collection):
                        gc.collect()
                    if(len(self.parent.custom_metrics)!=self.parent.n_custom_plots):
                        warnings.warn("Data provided does not match the number of custom plots")
                        self.parent.total_plots=len(self.parent.custom_metrics)
                        self.parent.n_custom_plots=len(self.parent.custom_metrics)
                    else:
                        self.parent.total_plots=self.parent.n_custom_plots

                    if((self.parent.top_rows)*(self.parent.top_cols)<self.parent.total_plots):
                        warnings.warn("Total number of plots does not match the number of rows, number of rows has been increased to \
                                      account for the difference")
                        while(((self.parent.top_rows)*(self.parent.top_cols)<self.parent.total_plots)):
                            self.parent.top_rows+=1

                    n_splits= self.parent.n_splits
    #                 len(cm_df_overall.index)//2


                    self.parent.figure = plt.figure(figsize=self.parent.figsize)
                    self.parent.main_grid = gridspec.GridSpec(self.parent.nrows,
                                                              self.parent.ncols,
                                                              hspace=self.main_grid_hspace,
                                                              wspace=self.main_grid_wspace)
                    self.parent.top_cell = self.parent.main_grid[0,0:]
                    self.parent.bottom_cell = self.parent.main_grid[1,0:]
                    self.parent.inner_grid_top = gridspec.GridSpecFromSubplotSpec(
                                                (self.parent.top_rows*self.parent.plot_width)+(self.parent.top_rows-1),
                                                (self.parent.top_cols*self.parent.plot_height)+(self.parent.top_cols-1),
                                                self.parent.top_cell,hspace=1
                                                )
                    self.parent.bottom_cell = self.parent.main_grid[1,0:]
                    self.parent.inner_grid_bottom = gridspec.GridSpecFromSubplotSpec(n_splits,3, self.parent.bottom_cell)


                    self.parent.top_axes=[]
                    for j,k in enumerate(range(0,self.parent.top_rows*self.parent.plot_width,self.parent.plot_width)):
                        for l,m in enumerate(range(0,self.parent.top_cols*self.parent.plot_height,self.parent.plot_height)):
                            temp=plt.subplot(self.parent.inner_grid_top[k+j:k+self.parent.plot_height+j,
                                                                        l+m:(l-1)+m+self.parent.plot_width])
                            self.parent.top_axes.append(temp)

                    self.parent._avg_axes=[]
                    for i,item in enumerate(self.parent.top_axes):
                        if(i<self.parent.total_plots):
                            if(self.parent.custom_metrics[i].average):
                                ax_avg=item.twiny().twinx()
                                self.parent._avg_axes.append(ax_avg)

    #                 self.parent.middle_cell = self.parent.main_grid[1,0:]
    #                 self.parent.inner_grid_middle = gridspec.GridSpecFromSubplotSpec(n_splits,3, self.parent.middle_cell)
    #                 middle_axes=[]
        #
        
                    #adds the main plots
                    for i in range(0,self.parent.n_custom_plots):
                        try:
                            if(self.parent.custom_metrics[i].updated):
                                custom_data=self.parent.custom_metrics[i].window()
                                if(custom_data.empty):
                                    custom_data=pd.DataFrame([0,0,0,0],columns=['No Data Available Yet'])
                                #This can be optimized later
                                
#                                 top_axes[i].clear()
                                if(not self.parent.custom_metrics[i].avg_only):
                                    self.parent.top_axes[i].plot(custom_data.iloc[-1*self.parent.custom_metrics[i].w_size:,:])
                                else:
                                    self.parent.top_axes[i].plot(custom_data)
                                self.parent.top_axes[i].legend(self.parent.custom_metrics[i].window().columns)
                                self.parent.top_axes[i].set_title(self.parent.custom_metrics[i].name)
                                self.parent.top_axes[i].set_ylabel('')
                                self.parent.top_axes[i].set_xlabel('')
                                if(self.parent.custom_metrics[i].xaxis_int):
                                    self.parent.top_axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                                #limit the number of ticks
                                if(self.parent.custom_metrics[i].n_ticks):
                                    _k,_l=self.parent.custom_metrics[i].n_ticks
                                    self.parent.top_axes[i].xaxis.set_major_locator(plt.MaxNLocator(_k,integer=True))
                                    self.parent.top_axes[i].yaxis.set_major_locator(plt.MaxNLocator(_l,integer=True))
#                                 self.parent.top_axes[i].set_xticks(self.parent.top_axes[i].get_xticks()[::2])
                                if(self.parent.custom_metrics[i].show_grid):
                                    item.grid()


                                if(self.parent.custom_metrics[i].average):
                                    self.parent._avg_axes[i].clear()
                                    avg=self.parent.custom_metrics[i].means
                                    self.parent._avg_axes[i].plot(avg,linestyle='--',alpha=0.6)
                                
                                    if(self.parent.custom_metrics[i].n_ticks):
                                        _k,_l=self.parent.custom_metrics[i].n_ticks
                                        self.parent._avg_axes[i].xaxis.set_major_locator(plt.MaxNLocator(_k,integer=True))
                                        self.parent._avg_axes[i].yaxis.set_major_locator(plt.MaxNLocator(_l,integer=True))                                        

                        except Exception as error:
                            print(error, 'Happened while adding main plots.')

                    # Adds table
                    split_df=[]
                    bottom_axes=[]
                    self.tail()
                    if(self.parent.show_table):
                        if(not self.parent.main_results.empty):
                            #round and limit the number of characters
                            _temp=self.parent.main_results.round(8).T.apply(lambda x:x.astype(str).str.slice(0,8))
                            for i,item in enumerate(self.chunks_df(_temp,
                                                                   math.ceil(len(self.parent.main_results.columns)/2),'r')):
                                if((i+1)%2==0 and (i>0)):
                                    loc='bottom right'
                                else:
                                    loc='bottom'
                                if(not item.empty):
                                    split_df.append(item)

                                    temp=plt.subplot(self.parent.inner_grid_bottom[i])
                                    temp.axis('off')
                                    temp.axis('tight')
                                    table=temp.table(cellText=item.values,
                                                loc=loc,
                                               cellLoc='left',
                                                rowLabels=item.index,
    #                                             colLabels=item.columns,
                                                )
                                    bottom_axes.append(table)
                                    table.scale(1, 2)



                            self.parent.split_df=split_df
                            self.parent.bottom_axes=bottom_axes


                    #moves legends to the bottom of the plots
                    for i,item in enumerate(self.parent.top_axes[:self.parent.n_custom_plots]):
                        box = item.get_position()
                        item.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
                        lines=item.get_lines()
                    #**** Following does not make sense
                        if(custom_data.empty):
                            custom_data=pd.DataFrame([0,0,0,0],columns=['No Data Available Yet'])

#                         lines[0].get_xydata()
                        item.legend(self.parent.custom_metrics[i].window().columns,loc='upper center',
                                    bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)


                            #add scientific formatting
        #                     item.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
  

            
                    #Removes empty plots
                    for i,ax in enumerate(self.parent.top_axes):
                        lines=ax.get_lines()
                        try:
                            lines[0].get_xydata()
                        except Exception as error:
                            self.parent.top_axes[i].axis('off')
                    plt.show()
                    if(self.parent.save_plots):
                        self.parent.figure.savefig(os.path.join(self.parent.image_folder,'image_'+str(self.parent.counter)+'.png'), 
                                                   bbox_inches = 'tight',pad_inches = .5,
                                                  )
                    for item in self.parent.custom_metrics:
                        if(item.updated):
                            item.updated=False
#                     plt.close(self.parent.figure)
                    clear_output(wait=True)
                    self.parent.counter+=1
        self._plot=plot(self,)
        self.plot=self._plot.create


class metric:
    def __init__(self,name=None,w_size=10,average=False,save_data=False,show_grid=False,xaxis_int=True,n_ticks=(3,3),
                avg_only=False):
        if(name):
            self.name=name
        else:
            raise Exception ('please provide a name for this metric.')
        self.data_folder=''
        self.save_data=save_data
        self.__kwargs=None
        self.counter=0
        self.keys=[]
        self.updated=False
        self.w_size=w_size+1
        self.last_chunk=pd.DataFrame()
        self.means=[]
        self.average=average
        self.n_ticks=n_ticks
        self.xaxis_int=xaxis_int
        if(not name):
            raise Exception('please provide a name for this metric.')
        self.show_grid=show_grid
        self.avg_only=avg_only
        if(self.avg_only):
            self.average=False
        self.save_data=save_data
        self.log_filename=''
        self.uid=uuid.uuid4().hex
            
    def __str__(self,):
        return ('metric')
    
    def update(self,**kwargs):
        self.updated=True
        self.counter+=1
        self.__kwargs=kwargs
        if(self.save_data):
            self.log_filename=self.uid
            log__(self.data_folder,{self.name:self.__kwargs},self.log_filename) 
        for key in self.__kwargs.keys():
            if(key in self.__dict__ ):
                self.__dict__[key].append(self.__kwargs[key])
            else:
                self.keys.append(key)
                self.__dict__[key]=[self.__kwargs[key]]
        if(len(self.__dict__[key])%self.w_size==0):
            self.chunk()
            self.chunk_mean()
            for key in self.keys:
                self.__dict__[key]=[]

    def frame(self,):
        _dict={}
        for k,v in self.__dict__.items():
            if(k in self.__dict__['keys']):
                _dict[k]=v
        _data=pd.DataFrame(_dict)
        if('x' in _data.columns):
            _data.set_index('x',inplace=True)
        return(_data)
    def window(self,):
        _data=pd.concat([self.last_chunk,self.frame()]).reset_index(drop=True)[(-1*self.w_size):]
        _data.index=list(range(1,self.counter+1,1))[(-1*self.w_size):]
        if('x' in _data.columns):
            _data.set_index('x',inplace=True)
        if(self.avg_only):
            if(self.means):
                _data=pd.concat(self.means,axis=1).T
            else:
                _data=pd.DataFrame([0,0,0,0],columns=['No Data Available Yet'])
        return _data

    def chunk(self,method='r'):
        _data=self.frame()
        n=self.w_size
        _output=[]
        if(method=='c'):
            for i in range(0, _data.shape[1], n):
                _output.append(_data.iloc[:,i:i + n])
        elif(method=='r'):
            for i in range(0, _data.shape[0], n):
                _output.append(_data.iloc[i:i + n,:])
        self.last_chunk=_output[0]
    def chunk_mean(self,):
        self.means.append(self.last_chunk.mean(axis=0))
    def shape(self,):
        return self.frame().shape
    
    def __getitem__(self,key):
        if(key in self.keys):
            _data=self.window()
            return(_data[key])
        else:
            return None
    def __len__(self,):
        return(len(self.keys))
    def __call__(self,):
        return self.frame().iloc[(-1*self.w_size):,:]
    
class pycmMetrics():
    def __init__(self,overall_metrics=[],class_metrics=[],name='',w_size=10,save_data=False):
        
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
        
        
        if(not name):
            raise Exception('please provide a name for the group metrics.')
        else:
            self.name=name
        self.save_data=save_data
        self.data_folder=''
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
        self.updated=False
        self.uid=uuid.uuid4().hex
        if(self.overall_metrics):
            for key in self.overall_metrics:
                if(key in self._overall_metrics):
                    _key=str(key).replace(' ','_')
                    self.metrics_oa[self.name+'_'+str(_key)]=metric(name=self.name+'_'+str(_key),w_size=self.w_size)
                    
        if(self.class_metrics):
            for key in self.class_metrics:
                if(key in self._class_metrics):
                    _key=str(key).replace(' ','_')
                    self.metrics_cls[self.name+'_'+str(_key)]=metric(name=self.name+'_'+str(_key),w_size=self.w_size)
        self.metrics=list({**self.metrics_oa,**self.metrics_cls}.values())

    def __str__(self,):
        return ('pycmMetrics')
    
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
        
    def _to_df(self,_cm):
        
        _main,_class=self._to_dict(_cm)
        self.cm_dict_overall=_main
        self.cm_dict_class=_class
        self.cm_df_overall=pd.DataFrame(_main,index=[0])
        self.cm_df_class=pd.DataFrame(_class,index=[0])

    def update(self,actual,predicted):
        self.updated=True
        _cm=ConfusionMatrix(actual,predicted)
        self._to_dict(_cm)
        if(self.save_data):
            self.log_filename=self.uid
            _log={self.name:{'overall_stat':_cm.__dict__['overall_stat'],'class_stat':_cm.__dict__['class_stat']}}
            log__(self.data_folder,_log,self.log_filename,) 
        if(self.metrics_oa):
            for k,v in self.cm_dict_overall.items():
                self.metrics_oa[self.name+'_'+str(k)].update(**{self.name+'_'+str(k):v})
        if(self.metrics_cls):
            for k,v in self.cm_dict_class.items():
                if(isinstance(v,dict)):
                    self.metrics_cls[self.name+'_'+str(k)].update(**{self.name+'_'+str(k)+'_'+str(k_1):v_1 for k_1,v_1 in v.items()})
                else:
                    self.metrics_cls[self.name+'_'+str(k)].update(**{k:v})
        self.metrics=list({**self.metrics_oa,**self.metrics_cls}.values())


class collate():
    def __init__(self,target_a,target_b,target_metric='',name=None,average=False,show_grid=False,xaxis_int=True,n_ticks=(3,3),
                avg_only=False):
        self.target=[target_a,target_b]
        if(str(target_a)!=str(target_b)):
            raise Exception ("Metric type/class mismatch.")
        else:
            if(not str(target_a) or not str(target_b)):
                raise Exception ("Metrics have no type/class")
        if(str(target_a) not in ['metric','pycmMetrics']):
            raise Exception ("type/class not recognized, has to be of class metric or pycmMetrics")
        if(str(target_a)=='metric'):
            self.type='metric'
        else:
            self.type='pycmMetrics'    
            
        if(self.type=='pycmMetrics'):
            self._all_metrics=list(set(target_a._all_metrics+target_b._all_metrics))
            if( target_metric in self._all_metrics):
                self.target_metric=str(target_metric).replace(' ','_')
            else:
                raise Exception ("Metric not found or is not available.")
        elif(self.type=='metric'):
            if(not target_a.avg_only or not target_a.avg_only):
                raise Exception ("Collate is only available to class metric when avg_only is True.")
            else:
                avg_only=True

        self.means=[]
        self.updated=False
        if(name):
            self.name=name
        else:
            if(target_metric):
                self.name=target_metric+' - '+target_a.name+' and '+target_b.name
            else:
                self.name=target_a.name+' and '+target_b.name
        if(target_a.w_size!=target_b.w_size):
            raise Exception ("Selected Metrics do not have the same w_size.")
        else:
            self.w_size=target_a.w_size
        self.average=average
        self.show_grid=show_grid
        self.xaxis_int=xaxis_int
        self.n_ticks=n_ticks
        self.avg_only=avg_only
    def update(self,):
        self.updated=True
        temp_a=[]
        if(self.type=='pycmMetrics'):
            for item_0 in self.target:
                for item_1 in item_0.metrics:
                    _key=item_1.name.replace(item_0.name+"_","")
                    if(_key==self.target_metric):
                        if(item_1.means):
                            temp_a.append(pd.concat(item_1.means, axis=1).T)
            if(temp_a):
                self.means=pd.concat(temp_a,axis=1)

    def window(self,):
        temp_a=[]
        if(self.type=='pycmMetrics'):
            for item_0 in self.target:
                for item_1 in item_0.metrics:
                    _key=item_1.name.replace(item_0.name+"_","")
                    if(_key==self.target_metric):
                        temp_a.append(item_1.window())
            self.update()
            return(pd.concat(temp_a,axis=1))
        elif(self.type=='metric'):
            if(not self.target[0].means or not self.target[1].means):
                return pd.DataFrame([0,0,0,0],columns=['No Data Available Yet'])
            else:
                return pd.concat([pd.concat(item.means,axis=1) for item in self.target],axis=0).T   

class metricsTable:
    def __init__(self,):
        self.counter=0
        self.updated=False
        self.__kwargs=None
        self.keys=[]
        self.uid=uuid.uuid4().hex
        
    def window(self,):
        _dict={}
        for k,v in self.__dict__.items():
            if(k in self.__dict__['keys']):
                _dict[k]=v
        _data=pd.DataFrame(_dict)
        return(_data)
        
    def update(self,**kwargs):
        self.updated=True
        self.counter+=1
        self.__kwargs=kwargs
        for key in self.__kwargs.keys():
            self.__dict__[key]=[self.__kwargs[key]]
            if(key not in self.keys):
                self.keys.append(key)            
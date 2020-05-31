from traintorch.utils import *

class Results():
    def __init__(self,project_name):
        self.__all_logs=[item for item in listall_ext(cwd,'.log')]
        self.__main_logs=[item  for item in self.__all_logs if "main.log" in item]
        if([item  for item in self.__all_logs if "main.log" not in item]):
            self.__run_logs=[item  for item in self.__all_logs if "main.log" not in item]
        else:
            self.__run_logs=[]
        if([getlognorm__(item) for item in self.__main_logs]):
            self.__all_runs=pd.concat([getlognorm__(item) for item in self.__main_logs]) 
        else:
            self.__all_runs=[]
        if(project_name):
            self.project_name=project_name
        else:
            self.project_name=''

    def ls_runs(self,project_name=''):
        if(not project_name):
            project_name=self.project_name
            return (self.__all_runs.loc[self.__all_runs.name==self.project_name]).to_dict('records')
        if(project_name):
            return (self.__all_runs.loc[self.__all_runs.name==self.project_name]).to_dict('records')
        else:
            return (self.__all_runs).to_dict('records')
    def ls_logs(self,project_name=''):
        if(not project_name):
            project_name=self.project_name
            return ([item for item in self.__run_logs if project_name in item])
        if(project_name):
            return ([item for item in self.__run_logs if project_name in item])
        else:
            return (self.__run_logs)

    def ls_checkpoints(self,project_name=''):
        if(not project_name):
            project_name=self.project_name
            return ([item for item in self.__run_logs if ((project_name in item) & ('checkpoints.log' in item))])
        if(project_name):
            return ([item for item in self.__run_logs if ((project_name in item) & ('checkpoints.log' in item))])
        else:
            return ([item for item in self.__run_logs if('checkpoints.log' in item)])

    def ls_configs(self,project_name=''):
        if([getlognorm__(item) for item in self.__main_logs]):
            main=pd.concat([getlognorm__(item) for item in self.__main_logs])
            cols=['name']+[item for item in main.columns if 'model_config' in item]
            if(not project_name):
                project_name=self.project_name
                output=main.loc[(main['project_name']==project_name)][cols].reset_index(drop=True)
                return output.set_index(['name']).T.to_dict()
            if(project_name):
                output=main.loc[(main['project_name']==project_name)][cols].reset_index(drop=True)
                return output.set_index(['name']).T.to_dict()
            else:
                output=pd.concat([getlognorm__(item) for item in self.__main_logs])[cols].reset_index(drop=True)
                return output.set_index(['name']).T.to_dict()
        else:
            return {}

    def ls_projects(self,project_name=''):
        if([getlognorm__(item) for item in self.__main_logs]):
            main=pd.concat([getlognorm__(item) for item in self.__main_logs])
            cols=['name']
            output={}
            if(not project_name):
                project_name=self.project_name
                result=list(main.loc[(main['project_name']==project_name)][cols].values.flatten())
                for item_r in result:
                    for item_c in self.__main_logs:
                         if(item_r in item_c):
                                output[item_r]=item_c
                return output
            if(project_name):
                result=list(main.loc[(main['project_name']==project_name)][cols].values.flatten())
                for item_r in result:
                    for item_c in self.__main_logs:
                         if(item_r in item_c):
                                output[item_r]=item_c
                return output
            else:
                result=list(pd.concat([getlognorm__(item) for item in self.__main_logs])[cols].values.flatten())
                for item_r in result:
                    for item_c in self.__main_logs:
                         if(item_r in item_c):
                                output[item_r]=item_c                
                return output
        else:
            return output 
    def checkpoints(self,columns='main'):
        checkpoints=self.ls_checkpoints()
        checkpoints_list=[]
        for checkpoint in checkpoints:

            specs=[]
            with open(checkpoint,"r") as F:
                for k,line in enumerate(F):
                    specs.append(json.loads(line))
            checkpoints_list.append(pd.io.json.json_normalize(specs))
        
        full=pd.concat(checkpoints_list)
        main=full.loc[:,full.columns.difference(['log_type', 'checkpoint_time', 'run_uid', 'run_time'])]
        if(columns=='all'):
            return full
        else:
            return main
        
    def runs(self,):    
        runs=self.ls_runs()
        return pd.DataFrame(runs)
    def metrics(self,):
        metric_logs=[item for item in self.ls_logs() if('checkpoint' not in item)]
        all_metrics=[]
        for item in self.ls_runs():
            metric_logs_list=[]
            for log in metric_logs:

                specs=[]
                with open(log,"r") as F:
                    for k,line in enumerate(F):
                        specs.append(json.loads(line))
                metric_logs_list.append(pd.io.json.json_normalize(specs))
            full=pd.concat(metric_logs_list) 
            full['run_uid']=item['uid']
            full['run_timestamp']=item['timestamp']
            full['run_project_name']=item['name']
            all_metrics.append(full)
        return pd.concat(all_metrics)
        
import sys
import os
import simplejson as json

cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)


def check_file(filename,location=cwd):    
    
    return os.path.exists(os.path.join(location,filename)),os.path.join(location,filename)

def check_folder(foldername,location=cwd):    
    
    return os.path.exists(os.path.join(location,foldername))

def create_folders(folders):
    for folder in folders:
        if(check_folder(folder)):
            pass
        else:
            os.mkdir(folder)
def to_log(location,log_filename,content):
        log_loc=os.path.join(location,log_filename)
        with open(log_loc, 'a') as f:
            if(os.stat(log_loc).st_size != 0):
                f.write('\n'+json.dumps(content))
            else:
                f.write(json.dumps(content))     
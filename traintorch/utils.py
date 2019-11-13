import sys
import os
import simplejson as json
import uuid
from uuid import UUID

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
def test_json(content={}):
    try:
        json.dumps(content)
        return True
    except:
        return False
    
def to_log(location,content,log_filename='',custom_name=False):
        try:
            temp=UUID(hex=log_filename,version=4)
        except:
            if(not custom_name):
                log_filename=uuid.uuid4().hex
        log_loc=os.path.join(location,log_filename+'.log')
        if(test_json(content)):
            with open(log_loc, 'a') as f:
                if(os.stat(log_loc).st_size != 0):
                    f.write('\n'+json.dumps(content))
                else:
                    f.write(json.dumps(content))
        else:
            raise Exception('Content not json serializable.') 
            
def find_checkpoints(folder,ext='.pth'):
    checkpoint_files=[]
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            for child_root, child_dirs, child_files in os.walk(dir):
                for filename in child_files:
                    if(filename.endswith(ext)):
                        if (os.path.join(folder,dir, filename) not in checkpoint_files):
                            checkpoint_files.append(os.path.join(folder,dir, filename))
    return(checkpoint_files)
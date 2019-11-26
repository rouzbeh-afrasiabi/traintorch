import sys
import os
import uuid
import re
import simplejson as json
from uuid import UUID
import shutil 
import numpy as np
from hashlib import md5

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
    
def log__(location,content,log_filename='',custom_name=False):
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

def snap__(destination='',default_extensions=[".py",".ipynb"]):
    
    global hashes
    hashes=[]
    ignore_dot=[item  for item in os.listdir(cwd) if(bool(re.match('^\.',item)))]
    ignore_underscore=[item  for item in os.listdir(cwd) if(bool(re.match('^\_',item)))]
    ignore_save=['save']
    defaultIgnore_folders=ignore_dot+ignore_underscore+ignore_save
    defaultIgnore_paths=[os.path.join(cwd,item) for item in defaultIgnore_folders]
    
    def make_archive(source, destination):
            base = os.path.basename(destination)
            name = base.split('.')[0]
            format = base.split('.')[1]
            archive_from = os.path.dirname(source)
            archive_to = os.path.basename(source.strip(os.sep))
            shutil.make_archive(name, format, archive_from, archive_to)
            shutil.move('%s.%s'%(name,format), destination)    

    def get_ignored(path,filenames):
        to_ignore=[]

        ignore_file='ttignore.txt'
        ttignore=[]
        if(os.path.exists(os.path.join(path,ignore_file))):
            with open(os.path.join(path,ignore_file),"r") as F:
                for k,line in enumerate(F):
                    ttignore.append(os.path.join(path,line.splitlines()[0]))
            if(ttignore):
                ttignore=list(np.hstack(np.array(ttignore))) 
        ignore_paths=defaultIgnore_paths+ttignore

        for filename in filenames:
            if(path in ignore_paths):
                to_ignore.append(filename)
            elif(any([filename.endswith(ext) for ext in default_extensions])):
                if(os.path.join(path,filename) in ignore_paths):
                    to_ignore.append(filename)
                else:
                    continue
            else:
                if(not os.path.isdir(filename)):
                    to_ignore.append(filename)
                else:
                    if(os.path.join(path,filename) in ignore_paths):
                        to_ignore.append(filename)
        return list(set(to_ignore))


    try:
        _temp=shutil.copytree(cwd,destination,ignore=get_ignored)
        base_name=os.path.dirname(os.path.normpath(destination))
        zip_file=os.path.join(base_name,'archive.zip')
        make_archive(destination,zip_file)
#         m = md5()
#         with open(os.path.join(destination,'archive.zip'), "rb") as f:
#             data = f.read()
#             m.update(data)
#             hashes.append(m.hexdigest())
    except:
        raise Exception ("Failed to create snapshot")   
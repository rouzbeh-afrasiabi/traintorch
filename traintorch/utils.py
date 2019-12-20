import sys
import os
import uuid
import re
import simplejson as json
from uuid import UUID
import shutil 
import numpy as np
from hashlib import md5
from checksumdir import dirhash

import nbformat
from nbconvert import HTMLExporter, PythonExporter
from nbconvert.writers import FilesWriter

import pandas as pd

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
def getlog__(log_file):
    content=[]
    with open(log_file,"r") as F:
        for k,line in enumerate(F):
            content.append(json.loads(line))
    return content

def getlognorm__(log_file):
    content=[]
    with open(log_file,"r") as F:
        for k,line in enumerate(F):
            content.append(json.loads(line))
    return pd.io.json.json_normalize(content)
            
def listall_ext(folder,ext='.pth'):
    checkpoint_files=[]
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if(filename.endswith(ext)):
                checkpoint_files.append(os.path.join(root, filename))
    return(list(set(checkpoint_files)))

def ipynb_to_py(target_folder):
    files=listall_ext(target_folder,'.ipynb')
    for file in files:
        file_name=os.path.join(os.path.dirname(file),os.path.basename(os.path.splitext(file)[0]))
        notebook_node=nbformat.read(file, as_version=4)
        exporter=PythonExporter()
        (body, resources) = exporter.from_notebook_node(notebook_node)
        write_file = FilesWriter()
        write_file.write(
            output=body,
            resources=resources,
            notebook_name=file_name
        )
        os.remove(file)

def snap__(destination='',default_extensions=[".py",".ipynb"]):
    
    ignore_dot=[item  for item in os.listdir(cwd) if(bool(re.match('^\.',item)))]
    ignore_underscore=[item  for item in os.listdir(cwd) if(bool(re.match('^\_',item)))]
    ignore_save=['save']
    defaultIgnore_folders=ignore_dot+ignore_underscore+ignore_save
    defaultIgnore_paths=[os.path.join(cwd,item) for item in defaultIgnore_folders]
      

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
        ipynb_to_py(os.path.join(cwd,destination))
        checksum=dirhash(destination,'md5')
        return (checksum)
    except:
        raise Exception ("Failed to create snapshot")  
        
def archive__(source, destination,filename):
    def make_archive(source, destination):
            #credit https://stackoverflow.com/users/155970/seanbehan
            base = os.path.basename(destination)
            name = base.split('.')[0]
            format = base.split('.')[1]
            archive_from = os.path.dirname(source)
            archive_to = os.path.basename(source.strip(os.sep))
            shutil.make_archive(name, format, archive_from, archive_to)
            shutil.move('%s.%s'%(name,format), destination)   
    zip_file=os.path.join(destination,filename+'.zip')
    make_archive(source,zip_file)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="traintorch",
    version="1.0.2",
    author="Rouzbeh Afrasiabi",
    author_email="rouzbeh.afrasiabi@gmail.com",
    description="Package for live visualization of model validation metrics during training of a machine learning model in jupyter notebooks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rouzbeh-afrasiabi/traintorch",
    download_url="https://github.com/rouzbeh-afrasiabi/traintorch/archive/v.1.0.2-alpha.tar.gz",
    install_requires=[
    'pandas==0.25.1',
    'pytest==5.2.1',
    'pycm==2.2',
    'setuptools==41.0.1',
    'matplotlib==3.1.1',
    'opencv_python==4.1.1.26',
    'simplejson==3.16.0',
    'ipython==7.8.0',
    'numpy==1.17.3',
     'imageio==2.5.0'
    ],
    keywords = ['training', 'visualization', 'loss','plot','live','jupyter notebook', 'matplotlib'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)

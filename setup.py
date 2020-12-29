#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:05:32 2020

@author: YANG Junjie
"""


from setuptools import setup,find_packages

setup(
   name='HVUS',
   version='1.0.6',
   description='HVUS calculator tool',
   author='YANG Junjie',
   author_email='junjie.yang@centalesupelec.fr',
   #packages=find_packages(exclude=('demo')),  
   packages=[""],
   #package_dir = {'': 'lib'},
   install_requires=['numpy', 'scipy'], 
)
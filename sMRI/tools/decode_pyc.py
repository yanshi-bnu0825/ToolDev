#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:28:29 2023

@author: yanshi
"""

import marshal
#__pycache__/cb_tools.cpython-38.pyc

s=open('../__pycache__/cb_tools.cpython-38.pyc','rb')
s.seek(8)

cb_tools=marshal.load(s)

exec(cb_tools)
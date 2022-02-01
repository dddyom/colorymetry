#!/home/dddyom/program_files/miniconda3/envs/colors/bin/python
 
 #-*- coding: utf-8  -*-
 
 
import cgi, cgitb

from Picture import main

storage = cgi.FieldStorage()
option = storage.getvalue('option')
img = storage.getvalue('img')
print('Status: 200 OK')
print('Content-Type: text/html')
print()

if img is not None:
    code_with_padding = f"{img}{'=' * (len(img) % 4)}"
    print(code_with_padding)
    result_base64_string = main(code_with_padding, option)
    print(result_base64_string)

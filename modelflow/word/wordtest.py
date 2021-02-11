# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:13:12 2021

@author: bruger
"""

import zipfile
import lxml.etree as etree
import xml.dom.minidom
import re

with open('frmltest2.docx','rb') as f :
    docx = zipfile.ZipFile(f)
    namelist = docx.namelist()
    content = docx.read('word/document.xml').decode('utf-8')
    uglyXml = xml.dom.minidom.parseString(docx.read('word/document.xml')).toprettyxml(indent='  ')

# x = etree.parse(content)
# print (etree.fromstring(content))


text_re = re.compile('>\n\s+([^<>\s].*?)\n\s+</', re.DOTALL)    
prettyXml = text_re.sub('>\g<1></', uglyXml)

print(prettyXml)

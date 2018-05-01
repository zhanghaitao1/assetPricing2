# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py


import sqlite3
conn=sqlite3.connect(r'E:\a\zotero.sqlite')

c=conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table';")

tablenames=c.fetchall()

tablenames

for tb in tablenames:
    c.execute("select * from {} LIMIT 5".format(tb[0]))
    print(tb[0],c.fetchall(),'\n\n')








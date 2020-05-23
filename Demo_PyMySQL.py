import pymysql
import pandas as pd

conn = pymysql.connect(host='192.168.56.101',
                       port=3306,
                       user ='dba', password = 'mysql',
                       db ='mimic3')
sql  = "select count(*) from ADMISSIONS;"
result = pd.read_sql_query(sql,conn)
print(result)
conn.close()
# 연결 성공!
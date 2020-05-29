import pymysql
from pandas import DataFrame as df

class DataSet(torch.utils.data.Dataset):
    def __init__(self, col_list=None, attr_list=None):
        self.col_default = ['ADMISSIONS', 'ICUSTAYS', 'INPUTEVENTS_MV', 'PATIENTS']
        self.attr_default = {'ADMISSIONS':['ADMISSION_TYPE'], 'ICUSTAYS':['LOS'], 'INPUTEVENTS_MV':['PATIENTWEIGHT'], 'PATIENTS':['DOB', 'DOD', 'DOD_HOSP', 'GENDER']}
        self.col_list = self.col_default
        self.attr_list = self.attr_default
        self.datasetX = None
        self.datasetY = None


    @property
    def getDateSet (self):
        col_list = self.col_list
        conn = pymysql.connect(host='192.168.56.104', user='dba', password='mysql', db='mimiciiiv14', charset='utf8')
        curs = conn.cursor(pymysql.cursors.DictCursor)
        # Select LABEVENTS,SUBJECT_ID FROM LABEVENTS JOIN PATIENTS on LABEVENTS.SUBJECT_ID = PATIENTS.SUBJECT_ID JOIN ADMISSIONS on PATIENTS.SUBJECT_ID = ADMISSIONS.SUBJECT_ID
        sql_line = 'SELECT'
        for col in col_list:
            for attr in self.attr_default[col]:
                sql_line += ' ,'+col+'.'+attr
        sql_line += ' FROM ' + col_list[0]
        # sql_line = list(sql_line)
        # sql_line[6] = ' '
        # "".join(sql_line)
        sql_line = sql_line[:7]+sql_line[8:]

        prev = col_list[0]
        for col in col_list[1:]:
            if col != 'PATIENTS':
                sql_line += ' JOIN {0} on {1}.SUBJECT_ID = {0}.SUBJECT_ID and {1}.HADM_ID = {0}.HADM_ID'.format(col, prev)
            else:
                sql_line += ' JOIN {0} on {1}.SUBJECT_ID = {0}.SUBJECT_ID'.format(col, prev)
            col_list[0] = col
        sql_line += ';'
        curs.execute(sql_line)
        result = curs.fetchall()
        self.datasetX = df(result)

        sql_line = 'SELECT SUBJECT_ID, ADMITTIME, DISCHTIME, DEATHTIME FROM ADMISSIONS;'
        curs.execute(sql_line)
        result = curs.fetchall()
        self.datasetY = df(result)

        self.datasetX = self.changeValue ()

        return self.datasetX, self.datasetY

    def setdataY (self):
        self.datasetY['YN'] = self.datasetY['DEATHTIME']
        self.datasetY['YN'] = self.datasetY['YN'][self.datasetY['YN']]
        return self.datasetY

    def changeValue (self):
        self.datasetX['LOS'] = self.datasetX.LOS.astype(int)
        self.datasetX['PATIENTWEIGHT'] = self.datasetX.PATIENTWEIGHT.astype(int)
        self.datasetX['GENDER'][self.datasetX['GENDER']=='F'] = 1
        self.datasetX['GENDER'][self.datasetX['GENDER'] == 'M'] = 0
        print(self.datasetX['LOS'])
        print(self.datasetX['PATIENTWEIGHT'])
        print(self.datasetX['GENDER'])
        return self.datasetX


list = ['MICROBIOLOGYEVENTS', 'LABEVENTS']
db = DataSet(col_list=list)
resultX, resultY = db.getDateSet
print(resultX)
print(resultY)


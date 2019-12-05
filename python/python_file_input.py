# python 文件数据读取

import pandas as pd  
df = pd.DataFrame([  
            ['green', 'M', 10.1, 'class1'],   
            ['red', 'L', 13.5, 'class2'],   
            ['blue', 'XL', 15.3, 'class1']])  
print (df)    
''' 数据集为以下内容，所有操作均对df进行
       0   1     2       3
0  green   M  10.1  class1
1    red   L  13.5  class2
2   blue  XL  15.3  class1
'''

print(df.loc[1])  #行为1

print(df.loc[0:1]) #行为0~1

print(df.loc[:,:]) #读取全部数据

print(df.iloc[0:1]) #读取行标签[0:1)的数据 前闭后开  行标签必须为int型

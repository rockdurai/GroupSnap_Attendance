import os
import numpy as np
dir='/home/santhosh/vs_code/dataset2'
files=os.listdir(dir)
print(files)
labels=[]
for i in files:
    x=(i.split('.')[0])[:4]
    if x=='sans':
        labels.append(0)
    elif x=='dura':
        labels.append(1)
    elif x=='sury':
        labels.append(2)
labels=np.asarray(labels)
np.save('labels',labels)


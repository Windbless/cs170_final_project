import numpy as np
def loadsparsedata(fn):
    
    fp = open(fn,"r")
    lines = fp.readlines()
    maxf = 0;
    T = [ ]
    count = 0
    for line in lines:
        line = line.rstrip('\n')
        f_list = [float(i) for i in line.split(" ") if i.strip()]
        T += f_list[1:11]
        for i in line.split()[1::1]:
            maxf+=1
    
    feature_wide = (int)(maxf/len(lines))
    X = np.reshape(T,(len(lines),feature_wide))
    #print(X.shape)
    #X = np.zeros((len(lines),feature))
    Y = np.zeros((len(lines),1))
    print(X.shape)
    print(Y.shape)
    for i, line in enumerate(lines):
        values = line.split()
        Y[i] = float(values[0])    
    return X,Y
(X,Y) = loadsparsedata("CS170_SMALLtestdata__80.txt")

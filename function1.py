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
def Forward_function(X,Y):
    (m,n) = X.shape# feature and label
    print("This dataset has "+ str(n) +" feature(not including the class attribute), with "+str(m)+" instances.\n")
def main_funtion():
    print("Welcome to Shiyao Feng feature selection Algorithm")
    #get_file_name = input("Type in the name of the file to test: ")
    print("Type in the name of the file to test: ")
    get_file_name = "CS170_SMALLtestdata__80.txt"
    (X,Y) = loadsparsedata(get_file_name)
    print("Type the number of the Algorithm you want to run.\n")
    print("\t1) Forward selection\n\t2) Backward Elimination\n\t3) shiyao's Speical Algorithm")
    selection_al = 1
    return X,Y,selection_al

# start here for main
X,Y,selection_al = main_funtion()
if(selection_al == 1):
    print("1) Forward selection")
    Forward_function(X,Y)
if(selection_al == 2):
    pass
if(selection_al == 3):
    pass








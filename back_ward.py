import numpy as np
import math
def point_distance(x,y):
    result = np.square((x-y)**2)
    return float(result.sum())
def loadsparsedata(fn):
    
    fp = open(fn,"r")
    lines = fp.readlines()
    maxf = 0;
    T = [ ]
    count = 0
    for line in lines:
        line = line.rstrip('\n')
        f_list = [float(i) for i in line.split(" ") if i.strip()]
        T += f_list[1:15]
        for i in line.split()[1::1]:
            maxf+=1
    
    feature_wide = (int)(maxf/len(lines))
    X = np.reshape(T,(len(lines),feature_wide))
    #print(X.shape)
    #X = np.zeros((len(lines),feature))
    Y = np.zeros((len(lines),1))
    for i, line in enumerate(lines):
        values = line.split()
        Y[i] = float(values[0])    
    return X,Y 
def main_funtion():
    print("Welcome to Shiyao Feng feature selection Algorithm")
    #get_file_name = input("Type in the name of the file to test: ")
    print("Type in the name of the file to test: ")
    #get_file_name = "shiyao_test.txt"
    get_file_name = "CS170_SMALLtestdata__110.txt"
    (X,Y) = loadsparsedata(get_file_name)
    print("Type the number of the Algorithm you want to run.\n")
    print("\t1) Forward selection\n\t2) Backward Elimination\n\t3) shiyao's Speical Algorithm")
    selection_al = 2
    return X,Y,selection_al
def backward_function(X,Y):
    current_list = [] # 0 to feature
    add_feature = np.arange(0, n, 1)
    current_list = add_feature.tolist()
    whole_accuracy = leave_one_out_validation(X,Y,whole_list,add_feature)
    print("Running nearest neighbor with all " 
        + str(n) + " features, using \"leaving-one-out\"evlation, I get an accuracy of "
        + str(whole_accuracy*100) + "%")
    Max_accuary = whole_accuracy
    return
X,Y,selection_al = main_funtion()
if(selection_al == 1):
    pass
if(selection_al == 2):
    print("2) Backward Elimination")
    backward_function(X,Y)
    #currentlist_for = backward_function(X,Y)
    #print("We have Forward_function to pick up feature is ", *currentlist_for)
if(selection_al == 3):
    pass











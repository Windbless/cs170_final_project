
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
    selection_al = 1
    return X,Y,selection_al

def leave_one_out_validation(X,Y,list_add):
    # current_list is [0 1 .. feature_wide] it is just mean feature 1 2 3 ...n
    # feature just add one number to currentlist
    (m,n) = X.shape# label and feature : n how many feature
    new_data_x = [] 
    for i in range (len(list_add)):
            if i == 0:
                new_data_x = X[:,list_add[i]]
                new_data_x = np.reshape(new_data_x,(m,1))
                #print("first",new_data_x)
            else:
                add = X[:,list_add[i]]
                add = np.reshape(add,(m,1))
                new_data_x = np.concatenate((new_data_x,add),axis=1)
                #print("after",new_data_x)
    #we calculate the whole accuarcy, it mean we use X data
    current_label = 0
    for i in range(m): # each line/row of distance of other line 
        mindistance = math.inf
        index_min_d = i
        for j in range(m): #each other line
            if i!=j:
                localdistance = float(point_distance(new_data_x[i],new_data_x[j]))
                if mindistance > localdistance:
                    mindistance = localdistance
                    index_min_d = j
        if Y[i] == Y[index_min_d]:
            current_label += 1
    accuracy = float(current_label) / m
    new_data_x = [] 
    return accuracy
X,Y,selection_al = main_funtion()
list_1 = [5,3]
list_2 = [5,8]

print(list_1[0]+1, list_1[1]+1,float(leave_one_out_validation(X,Y,list_1)))

print(list_2[0]+1,list_2[1]+1,float(leave_one_out_validation(X,Y,list_2)))

list_3 = [5,3,8]

print(list_3[0]+1,list_3[1]+1,list_3[2]+1,float(leave_one_out_validation(X,Y,list_3)))

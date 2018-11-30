import numpy as np
import math
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
def point_distance(x,y):
    result = np.square((x-y)**2)
    return float(result.sum())

def leave_one_out_validation(X,Y,list_here):
    # current_list is [0 1 .. feature_wide] it is just mean feature 1 2 3 ...n
    # feature just add one number to currentlist
    (m,n) = X.shape# label and feature : n how many feature 
    origin_X = np.arange(0, n, 1)
    
    accuracy = 0
    if set(list_here) == set(origin_X) : #we calculate the whole accuarcy, it mean we use X data
        current_label = 0
        for i in range(m): # each line/row of distance of other line 
            mindistance = math.inf
            index_min_d = i
            for j in range(m): #each other line
                if i!=j:
                    localdistance = float(point_distance(X[i],X[j]))
                    if mindistance > localdistance:
                        mindistance = localdistance
                        index_min_d = j
            if Y[i] == Y[index_min_d]:
                current_label += 1
        accuracy = current_label / m
        return accuracy
    curr_leb = 0
    new_data_x = []
    for i in range (len(list_here)):
            if i == 0:
                new_data_x = X[:,list_here[i]]
                new_data_x = np.reshape(new_data_x,(m,1))
            else:
                add = X[:,list_here[i]]
                add = np.reshape(add,(m,1))
                new_data_x = np.concatenate((new_data_x,add),axis=1)
    for i in range(m): # each line/row of distance of other line 
        mindistance = math.inf
        index_min_d = i
        for j in range(m): #each other line
            if i!=j:
                localdistance = point_distance(new_data_x[i],new_data_x[j])
                if mindistance > localdistance:
                    mindistance = localdistance
                    index_min_d = j
        if Y[i] == Y[index_min_d]:
            curr_leb += 1
    accuracy = curr_leb  / m
    return(accuracy)

def Forward_function(X,Y):
    depth = 0
    Max_accuary = 0
    (m,n) = X.shape# label and feature : n how many feature 
    print("This dataset has "+ str(n) 
        +" feature(not including the class attribute), with "+str(m)+" instances.\n")
    #next line is total accuracy for all feature
    
    whole_list = [] # 0 to feature
    add_feature = np.arange(0, n, 1)
    whole_list = add_feature.tolist()
    whole_accuracy = leave_one_out_validation(X,Y,whole_list)
    print("Running nearest neighbor with all " 
        + str(n) + " features, using \"leaving-one-out\"evlation, I get an accuracy of "
        + str(whole_accuracy*100) + "%")
    Max_accuary = whole_accuracy

    ### we need while loop to find the higher accuarcy
    currentlist = []
    origin_set = np.arange(0, n, 1) 
    remain_va = origin_set.tolist()  
    accuracy_f = 0
    ind = 0
    #print("remain_va",remain_va)
    for i in range(len(remain_va)):
        newlist = currentlist
        newlist.append(remain_va[i]) # add each remain from 0 to feature -1
        loca_accuracy = leave_one_out_validation(X,Y,newlist)
        if (loca_accuracy > accuracy_f ):
            accuracy_f = loca_accuracy
            ind = remain_va[i]
        newlist.remove(remain_va[i])
        print(str(remain_va[i]+1)," features accuracy loca is ", loca_accuracy)
        currentlist = []
    currentlist.append(ind)
    if Max_accuary < accuracy_f:
        Max_accuary = accuracy_f
    depth +=1
    print("we pick the No", str(depth),"feature is ", str(ind+1),"'s accuracy is ", Max_accuary*100,"%" )
    remain_va.remove(ind)
    stop = False 

    while not stop:
        stop = True
        depth +=1
        loca_high = 0
        loca_high_ind = 0
        for i in range(len(remain_va)):
            newlist = currentlist
            newlist.append(remain_va[i]) # add each remain from 0 to feature -1
            loca_accuracy = leave_one_out_validation(X,Y,newlist)
            if (loca_accuracy > loca_high ):
                loca_high = loca_accuracy
                loca_high_ind = remain_va[i]
            print(str(remain_va[i]+1)," features accuracy loca is ", loca_accuracy)
            newlist.remove(remain_va[i])
        if(loca_high > Max_accuary ):
            Max_accuary = loca_high
            ind = loca_high_ind
            stop = False
            print("we pick the No", str(depth),"feature is ", str(ind+1),"'s accuracy is ", Max_accuary* 100,"%")
            remain_va.remove(ind)
            currentlist.append(ind)
        else:
            print("we pick the No", str(depth),"feature is ", str(loca_high_ind+1),"'s accuracy is ", loca_accuracy* 100,"%")
            print("it is low than our max accuracy " , str(Max_accuary*100),"%")
            stop = True
            for i in range(len(currentlist)):
                currentlist[i] = currentlist[i]+1
            print("stop finding the accuracy because we can't find the higher accuracy, we have feature is ", 
                *currentlist," it's Max accuracy is ", Max_accuary*100,"%" )

    return currentlist

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

# start here for main
X,Y,selection_al = main_funtion()
currentlist_for = []
if(selection_al == 1):
    print("1) Forward selection")
    currentlist_for = Forward_function(X,Y)
    print("We have Forward_function to pick up feature is ", *currentlist_for)
if(selection_al == 2):
    pass
if(selection_al == 3):
    pass
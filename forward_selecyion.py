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
def distance(x,y):
    return ((x-y)**2).sum
def leave_one_out_validation_1D(matrix):
    (m,n) = matrix.shape
    corrent_number = 0
    for j in range (m-1):
        if j-1!=0:
            dl = abs(matrix[j][0] - matrix[j-1][0])
        if j != m:
            dr = abs(matrix[j][0]- matrix[j+1][0])
        if(dl <= dr):
            if(matrix[j-1][1] == matrix[j][1]):
                corrent_number +=1
        if(dr < dl):
            if(matrix[j+1][1] == matrix[j][1]):
                corrent_number +=1
    accuracy = corrent_number/m
    return accuracy
def make_add_feature_matirx(label,data,feature_add):
    (m,y) = label.shape
    list_x = data[:,feature_add]
    list_x_1 = list_x.reshape((m,1))
    fianl = np.concatenate((list_x_1,label),axis=1)
    x = fianl[fianl[:,0].argsort(kind='mergesort')]
    return x
def combin_cur_new_matrix(current_matix,new_matirx):
    new_matirx = np.concatenate((current_matix,new_matirx),axis=0)
    new_matirx_sort = new_matirx[new_matirx[:,0].argsort(kind='mergesort')]
    return new_matirx_sort
def Forward_function(X,Y):
    (m,n) = X.shape# feature and label
    print("This dataset has "+ str(n) 
        +" feature(not including the class attribute), with "+str(m)+" instances.\n")
    #next line is total accuracy for all feature

    list_1 = []
    list_2 = []
    count_1 = 0
    count_2 = 0
    for i in range (m):
        if(Y[i] == 1):
            list_1.append(X[i][6])
            count_1 +=1
        else:
            list_2.append(X[i][6])
            count_2 +=1
    whole_accuracy = 0
    print("Running nearest neighbor with all " 
        + str(n) + " features, using \"leaving-one-out\"evlation, I get an accuracy of "
        + str(whole_accuracy) + "%")
    currentset = []
    current_acc = np.zeros((n,1))
    max_number = 0
    max_index = 0
    for i in range(n):
        new_cur_matirx = make_add_feature_matirx(Y,X,i)
        current_acc[i] = leave_one_out_validation_1D(new_cur_matirx)
        if(current_acc[i]>max_number):
            max_number = current_acc[i]
            max_index = i+1
    print("feature " ,str(max_index), " accuracy is ", str(max_number))
    

    print(current_acc)
    #print(currentset) work
    
    return new_cur_matirx
   
    return
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
if(selection_al == 1):
    print("1) Forward selection")
    Forward_function(X,Y)
if(selection_al == 2):
    pass
if(selection_al == 3):
    pass
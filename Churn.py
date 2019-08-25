# Libraries used in the project
import numpy as np
from sklearn.svm import SVC, LinearSVC
import sklearn.preprocessing
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
import PIL.ImageTk
from tkinter import *
from tkinter.filedialog import askopenfilename
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


PATH = 'Final_Churn_DataSet.csv'

# Function below is used to read data from the csv file. The features will be stored in variable X and the output(ie churn) will be stored in y.
def Build_Data_Set():
    data_df = pd.read_csv(PATH, sep=',')
    train = data_df
    test = train.pop('churn')
    X = np.array(train.values)
    y = test.values.tolist()
    return X, y

# Function below is used to read data for Random Forest and Support Vector Machine Algorithms
def read_data():
    X, y = Build_Data_Set()
    return X, y

# Function used below will use Random Forest to perform prediction of the test data and try to predict accurate y values. It will also return the score of the classifier
def Analysis_RF(model, customer):
    predictions = model.predict(customer)
    if (model.predict_proba(customer)[0][0] < 0.60):
        predictions = 1
    return predictions

# Function used below will use SVM to perform prediction of the test data and try to predict accurate y values. It will also return the score of the classifier
def Analysis_SVM(model, customer):
    predictions = model.predict(customer)
    if (model.predict_proba(customer)[0][0] < 0.55):
        predictions = 1
    return predictions

# Function will perform classification using Random Forest.
def final_evalRF(customer):
    X,y=read_data()
    model = RandomForestClassifier(min_samples_leaf=4, min_samples_split=5, max_features='log2',
                                   class_weight='balanced', max_depth=25, random_state=0, n_estimators=25,
                                   criterion='entropy')
    model.fit(X, y)
    predicted = Analysis_RF(model, customer)
    return predicted

# Function will perform classification using Support Vector Machine with Polynomial Kernel
def final_evalSVM(customer):
    X,y=read_data()
    model = SVC(class_weight='balanced', kernel='poly', C=1, gamma=0.01, probability=True, degree=7)
    model.fit(X,y)
    predicted = Analysis_SVM(model, customer)
    return predicted

# Function to give the classfication report of Random Forest
def scoreOfRF(event):
    global resultString
    resultString.set("\t\nCLASSIFICATION REPORT (Random Forest) :\n\n     \t       \tPrecision\t Recall\t\tF1-score\t\tSupport\n\n0\t\t0.98           \t 0.97    \t\t0.97    \t\t1073    \n\n1\t\t0.82           \t 0.86    \t\t0.84    \t\t177     \n\nAvg/Total\t0.96           \t 0.95    \t\t0.95    \t\t1250    \n\n\nACCURACY SCORE IS : 0.9544")

# Function to give the classfication report of Random Forest
def scoreOfSVM(event):
    global resultString
    resultString.set("\t\nCLASSIFICATION REPORT (Support Vector Machine) :\n\n     \t       \tPrecision\t Recall\t\tF1-score\t\tSupport\n\n0\t\t0.97           \t 0.96    \t\t0.96    \t\t1073    \n\n1\t\t0.76           \t 0.81    \t\t0.78    \t\t177     \n\nAvg/Total\t0.94           \t 0.94    \t\t0.94    \t\t1250    \n\n\nACCURACY SCORE IS : 0.9368")

# Function to find whether a customer will churn or not using Random Forest
def churnPredictRF():
    global resultString
    global img
    if(str(entry11.get()) == "" or str(entry1.get()) == "" or str(entry2.get()) == "" or str(entry3.get()) == "" or str(entry4.get()) == "" or str(entry5.get()) == "" or str(entry6.get()) == "" or str(entry7.get()) == "" or str(entry8.get()) == "" or str(entry9.get()) == "" or str(entry10.get()) == "" or str(entry12.get()) == "" or str(entry13.get()) == "" or str(entry14.get()) == "" or str(entry15.get()) == "" or str(entry16.get()) == "" or str(entry17.get()) == ""):
        resultString.set("Please enter all the values!")
        evebutton.config(state="disabled")
        daybutton.config(state="disabled")
        nightbutton.config(state="disabled")
        twentyfourbutton.config(state="disabled")
        interbutton.config(state="disabled")
        customerbutton.config(state="disabled")
        interbutton.config(background=root.cget("background"))
        twentyfourbutton.config(background=root.cget("background"))
        customerbutton.config(background=root.cget("background"))
        daybutton.config(background=root.cget("background"))
        evebutton.config(background=root.cget("background"))
        nightbutton.config(background=root.cget("background"))
        img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
        labelImage.config(image=img)
    elif(float(entry11.get()) < 0 or float(entry1.get()) < 0 or float(entry2.get()) < 0 or float(entry3.get()) < 0 or float(entry4.get()) < 0 or float(entry5.get()) < 0 or float(entry6.get()) < 0 or float(entry7.get()) < 0 or float(entry8.get()) < 0 or float(entry9.get()) < 0 or float(entry10.get()) < 0 or float(entry12.get()) < 0 or float(entry13.get()) < 0 or float(entry14.get()) < 0 or float(entry15.get()) < 0  or float(entry16.get()) < 0 or float(entry17.get()) < 0):
        resultString.set("Please enter appropriate values!!!!")
        evebutton.config(state="disabled")
        daybutton.config(state="disabled")
        nightbutton.config(state="disabled")
        twentyfourbutton.config(state="disabled")
        interbutton.config(state="disabled")
        customerbutton.config(state="disabled")
        interbutton.config(background=root.cget("background"))
        twentyfourbutton.config(background=root.cget("background"))
        customerbutton.config(background=root.cget("background"))
        daybutton.config(background=root.cget("background"))
        evebutton.config(background=root.cget("background"))
        nightbutton.config(background=root.cget("background"))
        img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
        labelImage.config(image=img)
    elif((int(float(entry2.get())) != 0 and int(float(entry2.get())) != 1) or (int(float(entry3.get())) != 0 and int(
            float(entry3.get())) != 1)):
        resultString.set("Please enter 0 or 1 values for plans")
        evebutton.config(state="disabled")
        daybutton.config(state="disabled")
        nightbutton.config(state="disabled")
        twentyfourbutton.config(state="disabled")
        interbutton.config(state="disabled")
        customerbutton.config(state="disabled")
        interbutton.config(background=root.cget("background"))
        twentyfourbutton.config(background=root.cget("background"))
        customerbutton.config(background=root.cget("background"))
        daybutton.config(background=root.cget("background"))
        evebutton.config(background=root.cget("background"))
        nightbutton.config(background=root.cget("background"))
        img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
        labelImage.config(image=img)
    else:
        customerCluster = np.array([float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get()), float(entry5.get()),
                                    float(entry6.get()), float(entry7.get()), float(entry8.get()), float(entry9.get()), float(entry10.get()),
                                    float(entry11.get()), float(entry12.get()), float(entry13.get()), float(entry14.get()), float(entry15.get()), float(entry16.get()), float(entry17.get())])
        customerCluster = customerCluster.reshape(1,-1)
        customerPredict = np.array(
            [float(entry2.get()), float(entry3.get()), float(entry4.get()), float(entry5.get()),
             float(entry7.get()), float(entry8.get()), float(entry10.get()),
             float(entry11.get()), float(entry14.get()), float(entry15.get()),
             float(entry16.get()), float(entry17.get())])
        customerPredict = customerPredict.reshape(1, -1)
        zero_one = final_evalRF(customerPredict)
        if(zero_one == 1):
            clusterer(customerCluster)
            resultString.set("Random forest Classifier Result = " + str(zero_one)+"\n(Customer will Churn)")
        else:
            resultString.set("Customer will not Churn")
            evebutton.config(state="disabled")
            daybutton.config(state="disabled")
            nightbutton.config(state="disabled")
            twentyfourbutton.config(state="disabled")
            interbutton.config(state="disabled")
            customerbutton.config(state="disabled")
            interbutton.config(background=root.cget("background"))
            twentyfourbutton.config(background=root.cget("background"))
            customerbutton.config(background=root.cget("background"))
            daybutton.config(background=root.cget("background"))
            evebutton.config(background=root.cget("background"))
            nightbutton.config(background=root.cget("background"))
            img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
            labelImage.config(image=img)

# Function to find whether a customer will churn or not using Support Vector Machine
def churnPredictSVM():
    global img
    global resultString
    if (str(entry11.get()) == "" or str(entry1.get()) == "" or str(entry2.get()) == "" or str(
            entry3.get()) == "" or str(entry4.get()) == "" or str(entry5.get()) == "" or str(entry6.get()) == "" or str(
            entry7.get()) == "" or str(entry8.get()) == "" or str(entry9.get()) == "" or str(
            entry10.get()) == "" or str(entry12.get()) == "" or str(entry13.get()) == "" or str(
            entry14.get()) == "" or str(entry15.get()) == "" or str(entry16.get()) == "" or str(entry17.get()) == ""):
        resultString.set("Please enter all the values!!!!")
        evebutton.config(state="disabled")
        daybutton.config(state="disabled")
        nightbutton.config(state="disabled")
        twentyfourbutton.config(state="disabled")
        interbutton.config(state="disabled")
        customerbutton.config(state="disabled")
        interbutton.config(background=root.cget("background"))
        twentyfourbutton.config(background=root.cget("background"))
        customerbutton.config(background=root.cget("background"))
        daybutton.config(background=root.cget("background"))
        evebutton.config(background=root.cget("background"))
        nightbutton.config(background=root.cget("background"))
        img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
        labelImage.config(image=img)
    elif (float(entry11.get()) < 0 or float(entry1.get()) < 0 or float(entry2.get()) < 0 or float(
            entry3.get()) < 0 or float(entry4.get()) < 0 or float(entry5.get()) < 0 or float(entry6.get()) < 0 or float(
            entry7.get()) < 0 or float(entry8.get()) < 0 or float(entry9.get()) < 0 or float(
            entry10.get()) < 0 or float(entry12.get()) < 0 or float(entry13.get()) < 0 or float(
            entry14.get()) < 0 or float(entry15.get()) < 0 or float(entry16.get()) < 0 or float(entry17.get()) < 0):
        resultString.set("Please enter appropriate values!!!!")
        evebutton.config(state="disabled")
        daybutton.config(state="disabled")
        nightbutton.config(state="disabled")
        twentyfourbutton.config(state="disabled")
        interbutton.config(state="disabled")
        customerbutton.config(state="disabled")
        interbutton.config(background=root.cget("background"))
        twentyfourbutton.config(background=root.cget("background"))
        customerbutton.config(background=root.cget("background"))
        daybutton.config(background=root.cget("background"))
        evebutton.config(background=root.cget("background"))
        nightbutton.config(background=root.cget("background"))
        img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
        labelImage.config(image=img)
    elif ((int(float(entry2.get())) != 0 and int(float(entry2.get())) != 1) or (int(float(entry3.get())) != 0 and int(
            float(entry3.get())) != 1)):
        resultString.set("Please enter 0 or 1 values for plans")
        evebutton.config(state="disabled")
        daybutton.config(state="disabled")
        nightbutton.config(state="disabled")
        twentyfourbutton.config(state="disabled")
        interbutton.config(state="disabled")
        customerbutton.config(state="disabled")
        interbutton.config(background=root.cget("background"))
        twentyfourbutton.config(background=root.cget("background"))
        customerbutton.config(background=root.cget("background"))
        daybutton.config(background=root.cget("background"))
        evebutton.config(background=root.cget("background"))
        nightbutton.config(background=root.cget("background"))
        img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
        labelImage.config(image=img)
    else:
        customerCluster = np.array(
            [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get()), float(entry5.get()),
             float(entry6.get()), float(entry7.get()), float(entry8.get()), float(entry9.get()), float(entry10.get()),
             float(entry11.get()), float(entry12.get()), float(entry13.get()), float(entry14.get()),
             float(entry15.get()), float(entry16.get()), float(entry17.get())])
        customerCluster = customerCluster.reshape(1, -1)
        customerPredict = np.array(
            [float(entry2.get()), float(entry3.get()), float(entry4.get()), float(entry5.get()),
              float(entry7.get()), float(entry8.get()), float(entry10.get()),
             float(entry11.get()), float(entry14.get()),
             float(entry15.get()), float(entry16.get()), float(entry17.get())])
        data_df = pd.read_csv("F:\\study_related\\Final_year_project\\final_dataset\\Final_Churn_DataSetSVC.csv", sep=',')
        train = data_df
        test = train.pop('churn')
        X_svm_duplicate = np.array(train.values)
        y_duplicate = test.values.tolist()
        X_svm_duplicate = np.vstack([X_svm_duplicate,customerPredict])
        X_svm_duplicate = sklearn.preprocessing.minmax_scale(X_svm_duplicate,feature_range=(-6,6))
        customerPredict = X_svm_duplicate[5000]
        customerPredict = customerPredict.reshape(1, -1)
        zero_one = final_evalSVM(customerPredict)
        if (zero_one == 1):
            clusterer(customerCluster)
            resultString.set("Support vector machine Result = " + str(zero_one[0]) + "\n(Customer will Churn)")
        else:
            resultString.set("Customer will not Churn")
            evebutton.config(state="disabled")
            daybutton.config(state="disabled")
            nightbutton.config(state="disabled")
            twentyfourbutton.config(state="disabled")
            interbutton.config(state="disabled")
            customerbutton.config(state="disabled")
            interbutton.config(background=root.cget("background"))
            twentyfourbutton.config(background=root.cget("background"))
            customerbutton.config(background=root.cget("background"))
            daybutton.config(background=root.cget("background"))
            evebutton.config(background=root.cget("background"))
            nightbutton.config(background=root.cget("background"))
            img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
            labelImage.config(image=img)


# Function below is used to cluster the churning customer's data and match is to the nearest non churning customer's data. Also contains GUI components
def clusterer(cust_in):
    global img
    evebutton.config(state="normal")
    daybutton.config(state="normal")
    nightbutton.config(state="normal")
    interbutton.config(state="normal")
    customerbutton.config(state="normal")
    twentyfourbutton.config(state="disabled")
    interbutton.config(background=root.cget("background"))
    customerbutton.config(background=root.cget("background"))
    twentyfourbutton.config(background=root.cget("background"))
    daybutton.config(background=root.cget("background"))
    evebutton.config(background=root.cget("background"))
    nightbutton.config(background=root.cget("background"))
    data_df_key = pd.read_csv(
        "Final_Churn_DataSet_s.csv",sep=',')
    hashing = dict()
    dfkey = data_df_key.values;
    tuple_key = tuple(dfkey)
    a = data_df_key.as_matrix()
    kmeans = KMeans(n_clusters=8,random_state=42)
    kmeans.fit(a)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels = np.array(labels)
    a = np.array(a)
    centroid_hash = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 0: []}
    for i in range(0, len(labels)):
        centroid_hash[labels[i]].append((a[i]))
    centroids = np.array(centroids)
    customer = cust_in[0]
    customer = np.array(customer)
    minimum = 99999
    dis = 0
    distance_array = []
    for i in range(0, len(centroids)):
        for j in range(0, len(customer)):
            currentcentroid = centroids[i]
            dis = dis + pow(customer[j] - currentcentroid[j], 2)
        dis = pow(dis, 0.5)
        if dis < minimum:
            minimum = dis
        distance_array.append(dis)
        dis = 0
    hashkey = distance_array.index(minimum)
    cluster_values = np.array(centroid_hash[distance_array.index(minimum)])
    centroid_cluster = centroids[distance_array.index(minimum)]
    minimum = 99999
    dis = 0
    distance_array = []
    for i in range(0, len(cluster_values)):
        for j in range(0, len(centroid_cluster)):
            current = cluster_values[i]
            dis = dis + pow(centroid_cluster[j] - current[j], 2)
        dis = pow(dis, 0.5)
        if dis < minimum:
            minimum = dis
        distance_array.append(dis)
        dis = 0
    hashvalue = centroid_hash[hashkey]
    nonChurner = hashvalue[distance_array.index(minimum)]
    nonChurner = np.array(nonChurner).tolist()
    Day_minute_plan = 0
    Day_minute_doubt = 0
    Evening_minute_plan = 0
    Evening_minute_doubt = 0
    Night_minute_plan = 0
    Night_minute_doubt = 0
    International_plan = 0
    Service_plan = 0
    flag = 0
    if (customer[4]>233.8):

        if(customer[4]/nonChurner[4] > 278.5/224.3):
            daybutton.config(bg="#FF0000")
            Day_minute_plan = 1
            dflag=1
        else:
            daybutton.config(bg="#FFA500")
            Day_minute_doubt = 1
    if (customer[7] > 250.6):

        if (customer[7] / nonChurner[7] > 263.5 / 247.9):
            evebutton.config(bg="#FF0000")
            Evening_minute_plan = 1
            dflag = 1
        else:
            evebutton.config(bg="#FFA500")
            Evening_minute_doubt = 1
    if (customer[10] > 250.4):

        if (customer[10] / nonChurner[10] > 256.3 / 249.2):
            nightbutton.config(bg="#FF0000")
            Night_minute_plan = 1
            dflag = 1
        else:
            nightbutton.config(bg="#FFA500")
            Night_minute_doubt = 1
    if (customer[1] == 0 and customer[13] > 10.23):
        International_plan = 1
        interbutton.config(bg="#FF0000")
        flag = 1
    elif (customer[13] > nonChurner[13] and customer[1] == 1):
        if (nonChurner[1] == 1):
            International_plan = 1
            interbutton.config(bg="#FF0000")
            flag = 1
    if (customer[16] > 3):
        Service_plan = 1
        customerbutton.config(bg="#FF0000")
        flag = 1
    if(Evening_minute_plan == 1 and Night_minute_plan == 1 and Day_minute_plan == 1):
            Day_minute_plan = 1
            twentyfourbutton.config(state="normal")
            twentyfourbutton.config(bg="#FF0000")
            daybutton.config(background=root.cget("background"))
            evebutton.config(background=root.cget("background"))
            nightbutton.config(background=root.cget("background"))
            daybutton.config(state="disabled")
            evebutton.config(state="disabled")
            nightbutton.config(state="disabled")
    if(Day_minute_plan == 0 and Day_minute_doubt == 0):
        daybutton.config(bg="#5BFF66")
    if(Evening_minute_plan == 0 and Evening_minute_doubt == 0):
        evebutton.config(bg="#5BFF66")
    if (Night_minute_plan == 0 and Night_minute_doubt == 0):
         nightbutton.config(bg="#5BFF66")
    if (International_plan == 0):
         interbutton.config(bg="#5BFF66")
    if (Service_plan == 0):
        customerbutton.config(bg="#5BFF66")

    activitySet = ['Day Mins', 'Evening Mins', 'Night Mins', 'International Plan', 'Customer Service']
    if(International_plan == 1):
        interImp = 10
    else:
        interImp = 0
    if(Day_minute_plan == 1 ):
        dayImp = 29.2
    elif(Day_minute_doubt == 1):
        dayImp = 29.2/2
    else:
        dayImp = 0
    if(Evening_minute_plan == 1):
        eveImp = 13
    elif(Evening_minute_doubt == 1):
        eveImp = 13/2
    else:
        eveImp = 0
    if(Night_minute_plan == 1):
        nightImp = 9.3
    elif(Night_minute_doubt == 1):
        nightImp = 9.3/2
    else:
        nightImp = 0
    if(Service_plan == 1):
        custImp = 15.1
    else:
        custImp = 0
    plans = [dayImp, eveImp, nightImp, interImp, custImp]
    col = ['CadetBlue','LightPink','yellow','RosyBrown','tomato']
    slices = []
    colors = []
    activities = []
    sliceslen = 0
    plt.rcParams['font.size'] = 9
    for i in range(0, len(col)):
        if (plans[i] != 0):
            slices.append(plans[i])
            colors.append(col[i])
            activities.append(activitySet[i])
    plt.pie(slices, labels=activities, colors=colors,
        startangle=90, shadow=True,
        radius=0.8, autopct='%1.1f%%')
    plt.savefig("churn.jpg")
    img = PIL.ImageTk.PhotoImage(PIL.Image.open("churn.jpg"))
    labelImage.config(image = img)
    plt.gcf().clear()


# Function to create GUI components
def dialogBox():
    root.fileName = askopenfilename(initialdir = "\\",title = " choose the CSV file containing the Customer Data",filetypes = (("CSV files","*.csv"),("all files","*.*")))
    csvFile = root.fileName
    try:
        inputCSV = pd.read_csv(csvFile, sep=',')
    except FileNotFoundError:
        print("File not found")
    else:
        entry1.delete(0,"end")
        entry1.insert(0,inputCSV.values[0][0])
        entry2.delete(0, "end")
        entry2.insert(0, inputCSV.values[0][1])
        entry3.delete(0, "end")
        entry3.insert(0, inputCSV.values[0][2])
        entry4.delete(0, "end")
        entry4.insert(0, inputCSV.values[0][3])
        entry5.delete(0, "end")
        entry5.insert(0, inputCSV.values[0][4])
        entry6.delete(0, "end")
        entry6.insert(0, inputCSV.values[0][5])
        entry7.delete(0, "end")
        entry7.insert(0, inputCSV.values[0][6])
        entry8.delete(0, "end")
        entry8.insert(0, inputCSV.values[0][7])
        entry9.delete(0, "end")
        entry9.insert(0, inputCSV.values[0][8])
        entry10.delete(0, "end")
        entry10.insert(0, inputCSV.values[0][9])
        entry11.delete(0, "end")
        entry11.insert(0, inputCSV.values[0][10])
        entry12.delete(0, "end")
        entry12.insert(0, inputCSV.values[0][11])
        entry13.delete(0, "end")
        entry13.insert(0, inputCSV.values[0][12])
        entry14.delete(0, "end")
        entry14.insert(0, inputCSV.values[0][13])
        entry15.delete(0, "end")
        entry15.insert(0, inputCSV.values[0][14])
        entry16.delete(0, "end")
        entry16.insert(0, inputCSV.values[0][15])
        entry17.delete(0, "end")
        entry17.insert(0, inputCSV.values[0][16])


# Main function to create GUI components and run the models on the data
def main():
    X, y = read_data()
    X_svm, y_svm = read_data()
    root = Tk()
    root.title("Churn prediction in Telecommunication Industry")
    frameInput = Frame(root, borderwidth=4, relief="raise")
    frameInput.grid(row=0, column=0, columnspan=4, rowspan=4, padx=6, pady=6)
    Label(frameInput, text="Enter customer data").grid(row=0, column=0, columnspan=5, padx=5, pady=5)
    frameInputOne = Frame(frameInput, borderwidth=3, relief="raise")
    frameInputOne.grid(row=1, column=0, padx=6, pady=6)
    frameInputTwo = Frame(frameInput, borderwidth=3, relief="raise")
    frameInputTwo.grid(row=1, column=1, padx=6, pady=6, sticky="n")

    Label(frameInputOne, text="Account Length").grid(row=0, column=0, padx=5, pady=5)
    entry1 = Entry(frameInputOne, width=20)
    entry1.grid(row=0, column=1, padx=5, pady=5)

    Label(frameInputOne, text="International Plan").grid(row=1, column=0, padx=5, pady=5)
    entry2 = Entry(frameInputOne, width=20)
    entry2.grid(row=1, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Voice Mail Plan").grid(row=2, column=0, padx=5, pady=5)
    entry3 = Entry(frameInputOne, width=20)
    entry3.grid(row=2, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Number of Voice Mail Messages").grid(row=3, column=0, padx=5, pady=5)
    entry4 = Entry(frameInputOne, width=20)
    entry4.grid(row=3, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Total day minutes").grid(row=4, column=0, padx=5, pady=5)
    entry5 = Entry(frameInputOne, width=20)
    entry5.grid(row=4, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Total day calls").grid(row=5, column=0, padx=5, pady=5)
    entry6 = Entry(frameInputOne, width=20)
    entry6.grid(row=5, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Total day charge").grid(row=6, column=0, padx=5, pady=5)
    entry7 = Entry(frameInputOne, width=20)
    entry7.grid(row=6, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Total evening minutes").grid(row=7, column=0, padx=5, pady=5)
    entry8 = Entry(frameInputOne, width=20)
    entry8.grid(row=7, column=1, padx=5, pady=5)

    Label(frameInputOne, text="Total evening calls").grid(row=8, column=0, padx=5, pady=5)
    entry9 = Entry(frameInputOne, width=20)
    entry9.grid(row=8, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total evening charge").grid(row=0, column=0, padx=5, pady=5)
    entry10 = Entry(frameInputTwo, width=20)
    entry10.grid(row=0, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total night minutes").grid(row=1, column=0, padx=5, pady=5)
    entry11 = Entry(frameInputTwo, width=20)
    entry11.grid(row=1, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total night calls").grid(row=2, column=0, padx=5, pady=5)
    entry12 = Entry(frameInputTwo, width=20)
    entry12.grid(row=2, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total night charge").grid(row=3, column=0, padx=5, pady=5)
    entry13 = Entry(frameInputTwo, width=20)
    entry13.grid(row=3, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total international minutes").grid(row=4, column=0, padx=5, pady=5)
    entry14 = Entry(frameInputTwo, width=20)
    entry14.grid(row=4, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total international calls").grid(row=5, column=0, padx=5, pady=5)
    entry15 = Entry(frameInputTwo, width=20)
    entry15.grid(row=5, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Total international charge").grid(row=6, column=0, padx=5, pady=5)
    entry16 = Entry(frameInputTwo, width=20)
    entry16.grid(row=6, column=1, padx=5, pady=5)

    Label(frameInputTwo, text="Number of customer service calls").grid(row=7, column=0, padx=5, pady=5)
    entry17 = Entry(frameInputTwo, width=20)
    entry17.grid(row=7, column=1, padx=5, pady=5)

    retrieve = Button(frameInputTwo, text="Retrieve CSV", width=20, fg="blue", command=dialogBox)
    retrieve.grid(row=8, column=0, columnspan=5, padx=5, pady=5)

    frameClassifier = Frame(root, borderwidth=4, relief="raise")
    frameClassifier.grid(row=4, column=0, columnspan=4, rowspan=1, padx=6, pady=6)
    frameSVM = Frame(frameClassifier, borderwidth=3, relief="raise")
    frameSVM.grid(row=0, column=0, padx=5, pady=5)
    frameRF = Frame(frameClassifier, borderwidth=3, relief="raise")
    frameRF.grid(row=0, column=1, padx=5, pady=5)

    scoreSVM = Button(frameSVM, text="Score", width=20)
    scoreSVM.bind("<Button-1>", scoreOfSVM)
    scoreSVM.grid(row=1, column=0, padx=5, pady=5)
    Label(frameSVM, text="Support Vector Machine").grid(row=0, column=0, columnspan=5, padx=0, pady=0)
    predictSVM = Button(frameSVM, text="Prediction", width=20, command=churnPredictSVM)
    # predictSVM.bind("<Button-1>",churnPredictSVM)
    predictSVM.grid(row=1, column=1, padx=5, pady=5)

    scoreRF = Button(frameRF, text="Score", width=20)
    scoreRF.bind("<Button-1>", scoreOfRF)
    scoreRF.grid(row=1, column=0, padx=5, pady=5)
    Label(frameRF, text="Random Forest Classifier").grid(row=0, column=0, columnspan=5, padx=0, pady=0)
    predictRF = Button(frameRF, text="Prediction", width=20, command=churnPredictRF)
    # predictRF.bind("<Button-1>",churnPredictRF)
    predictRF.grid(row=1, column=1, padx=5, pady=5)

    frameResult = Frame(root, borderwidth=4, relief="sunken", bg="white")
    frameResult.grid(row=5, column=0, columnspan=4, rowspan=1, padx=5, pady=5)
    resultString = StringVar()
    labelResult = Label(frameResult, textvariable=resultString, width=95, height=15, bg="white")
    labelResult.grid(row=0, column=0, padx=5, pady=5)

    framePlan = Frame(root, borderwidth=4, relief="raise", width=200)
    framePlan.grid(row=0, column=5, rowspan=1, columnspan=4, padx=5, pady=10, sticky="n")
    Label(framePlan, text="---Scope for Improvement---").grid(row=0, column=0, columnspan=5, padx=5, pady=5)
    daybutton = Button(framePlan, text="Day minute plan", width=25, relief="groove", state="disabled")
    daybutton.grid(row=1, column=0, padx=5, pady=5)
    evebutton = Button(framePlan, text="Evening minute plan", width=25, relief="groove", state="disabled")
    evebutton.grid(row=1, column=1, padx=5, pady=5)
    nightbutton = Button(framePlan, text="Night minute plan", width=25, relief="groove", state="disabled")
    nightbutton.grid(row=1, column=2, padx=5, pady=5)
    interbutton = Button(framePlan, text="International plan", width=25, relief="groove", state="disabled")
    interbutton.grid(row=2, column=0, padx=5, pady=5)
    customerbutton = Button(framePlan, text="Improve customer service", width=25, relief="groove", state="disabled")
    customerbutton.grid(row=2, column=1, padx=5, pady=5)
    twentyfourbutton = Button(framePlan, text="24hr plan", width=25, relief="groove", state="disabled")
    twentyfourbutton.grid(row=2, column=2, padx=5, pady=5)

    testFrame = Frame(framePlan)
    testFrame.grid(row=3, column=0, columnspan=9, padx=5, pady=5)
    Label(testFrame, text="   ", bg="#FF0000").grid(row=0, column=0, pady=5)
    Label(testFrame, text=" High Priority").grid(row=0, column=1, pady=5, sticky="w")
    Label(testFrame, text="   ", bg="#FFA500").grid(row=1, column=0, pady=5)
    Label(testFrame, text=" Medium Priority").grid(row=1, column=1, pady=5, sticky="w")
    Label(testFrame, text="   ", bg="#5BFF66").grid(row=2, column=0, pady=5)
    Label(testFrame, text=" Low Priority").grid(row=2, column=1, pady=5, sticky="w")

    frameImage = Frame(root, borderwidth=4, relief="sunken", bg="white")
    frameImage.grid(row=1, column=5, rowspan=20, columnspan=4, padx=5, pady=3, sticky="n")
    img = PIL.ImageTk.PhotoImage(PIL.Image.open("white.jpg"))
    labelImage = Label(frameImage, image=img, width=600, height=430, bg="white")
    labelImage.grid(row=0, column=0, padx=5, pady=5)
    root.mainloop()

main()

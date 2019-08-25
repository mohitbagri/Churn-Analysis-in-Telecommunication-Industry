import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
import pandas as pd
import pydot
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile
import pydotplus
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import seaborn as sns
import matplotlib

PATH = '\\Users\\MOHIT\\Desktop\\FINAL FINAL PROJECT\\FINAL YEAR\\Not Churn Data.csv'
NF = 8
PATH2='\\Users\\MOHIT\\Desktop\\FINAL FINAL PROJECT\\FINAL YEAR\\Final_Churn_Dataset.csv'

def graphs():
    data = pd.read_csv(PATH2, sep=',')

    CustServ_Calls = data.groupby('number_customer_service_calls').size()
    data["number_customer_service_calls"].hist(bins=500, figsize=(12, 10))
    plt1.title("NUMBER OF SERVICE CALLS ",fontsize=30)
    plt1.xlabel('CUSTOMER SERVICE CALLS', fontsize=18)
    plt1.ylabel('COUNT OF CUSTOMER SERVICE CALLS', fontsize=18)
    plt1.show()

    plt.title("HISTOGRAM OF DAY MINUTES",fontsize=25)
    Account_Length = data["total_day_minutes"]
    matplotlib.pyplot.hist(Account_Length, bins=500)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()

    plt.title("HISTOGRAM OF EVENING MINUTES", fontsize=25)
    Account_Length = data["total_eve_minutes"]
    matplotlib.pyplot.hist(Account_Length, bins=500)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()

    plt.title("HISTOGRAM OF NIGHT MINUTES", fontsize=25)
    Account_Length = data["total_night_minutes"]
    matplotlib.pyplot.hist(Account_Length, bins=500)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()

    Intl_Churn = data.groupby(["international_plan", 'churn']).size()
    Churn = data.groupby('churn').size()
    Intl_Churn.plot()
    #print(Intl_Churn)

    plt.xlabel("INTERNATION PLAN WISE CHURN", fontsize=18)
    plt.ylabel("COUNT OF INTERNATION PLAN WISE CHURN", fontsize=18)
    plt.show()

    sns.set_style("whitegrid")
    ax = sns.boxplot(x="international_plan", y="total_intl_minutes", hue="churn", data=data, palette="Set1")
    plt.title("BOXPLOT OF TOTAL INTERNATIONAL MINUTES VS INTERNATIONAL PLAN",size=24)
    plt.setp(ax.get_legend().get_texts(), fontsize='20')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='20')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)
    plt.xlabel('INTERNATIONAL PLAN', fontsize=24)
    plt.ylabel('TOTAL INTERNATIONAL MINUTES', fontsize=22)
    plt.show()

    g = sns.factorplot(y="account_length", x="churn", data=data,
                       size=6, kind="box", palette="Set1")
    plt.title("BOXPLOT OF ACCOUNT LENGTH VS CHURN")
    plt.show()

    g = sns.factorplot(y="total_day_minutes", x="churn", data=data,
                       size=6, kind="box", palette="Set1")
    plt.title("BOXPLOT OF TOTAL DAY MINUTES VS CHURN",fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('CHURN', fontsize=24)
    plt.ylabel('TOTAL DAY MINUTES',fontsize=24)
    plt.show()

    g = sns.factorplot(y="total_day_charge", x="churn", data=data,
                       size=4, kind="box", palette="Set1")
    plt.title("BOXPLOT OF TOTAL DAY CHARGE VS CHURN")
    plt.show()

    g = sns.factorplot(y="total_eve_minutes", x="churn", data=data,
                       size=6, kind="box", palette="Set1")
    plt.title("BOXPLOT OF TOTAL EVENING MINUTES VS CHURN",fontsize=24)
    plt.xlabel('CHURN', fontsize=24)
    plt.ylabel('TOTAL EVENING MINUTES', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

    g = sns.factorplot(y="total_night_minutes", x="churn", data=data,
                       size=6, kind="box", palette="Set1")
    plt.title("BOXPLOT OF TOTAL NIGHT MINUTES VS CHURN",fontsize=24)
    plt.xlabel('CHURN', fontsize=24)
    plt.ylabel('TOTAL NIGHT MINUTES', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

    VMail_Churn = data.groupby(['voice_mail_plan', 'churn']).size()
    VMail_Churn.plot()
    plt.title("NUMBER OF VOICE MAILS")
    plt.xlabel('VOICE MAIL WISE CHURN', fontsize=18)
    plt.ylabel('COUNT OF VOICE MAIL WISE CHURN', fontsize=18)
    plt.show()

    Custserv_Chrun = data.groupby(['number_customer_service_calls', 'churn']).size()
    Custserv_Chrun.plot(kind='bar', figsize=(10, 8))
    plt.xlabel('CUSTOMER SERVICE CALLS WISE CHURN', fontsize=18)
    plt.ylabel('COUNT OF CUSTOMER SERVICE CALLS WISE CHURN', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

graphs()

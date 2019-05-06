import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("diabetes.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, 8].values

clf = tree.DecisionTreeClassifier(criterion='entropy')
gnb = GaussianNB()
knn=KNeighborsClassifier(n_neighbors=5)
knn=knn.fit(x,y)
clf = clf.fit(x, y)
gnb=gnb.fit(x, y)
X=np.array(pd.read_csv('d2.csv'))
kmeans = KMeans(n_clusters=2, random_state=0, init=np.array([[0,0],[189,846]]), n_init=1).fit(X)

#5 , 116, 74, 2, 4, 25.6, 0.201, 30
#0	148	76	12	150	33.6	0.627	50
#0	148	76	12	120	33.6	0.627	50
#0	130	76	12	140	33.6	0.627	50
from tkinter import *
from PIL import ImageTk, Image


root=Tk()
root.title("Diabetes Predictor")
root.geometry("600x450")

def acc():
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    clf1 = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)
    gnb1 = GaussianNB().fit(X_train,y_train)
    knn1=KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
    y1pred=clf1.predict(X_test)
    y2pred=gnb1.predict(X_test)
    y3pred=knn1.predict(X_test)
    
    e1=accuracy_score(y_test,y1pred)
    e2=accuracy_score(y_test,y2pred)
    e3=accuracy_score(y_test,y3pred)
    
    a1.configure(state='normal')
    a1.delete(0, END)
    s1="Decision Tree Accuracy: "+str(round(e1*100,2))+"%"
    a1.insert(END,s1)
    a1.configure(state='disabled')
    
    a2.configure(state='normal')
    a2.delete(0, END)
    s2="Naive Bayes Accuracy: "+str(round(e2*100,2))+"%"
    a2.insert(END,s2)
    a2.configure(state='disabled')
    
    a3.configure(state='normal')
    a3.delete(0, END)
    s3="KNN Accuracy: "+str(round(e3*100,2))+"%"
    a3.insert(END,s3)
    a3.configure(state='disabled')
    
    
    
    
def action():
    preg=int(ip2.get())
    glu=int(ip3.get())
    blood=int(ip4.get())
    skin=int(ip5.get())
    insulin=int(ip6.get())
    bmi=float(ip7.get())
    pdf=float(ip8.get())
    age=int(ip9.get())
    
    ans1=clf.predict([[preg , glu, blood, skin, insulin, bmi, pdf, age]])
    ans2=gnb.predict([[preg , glu, blood, skin, insulin, bmi, pdf, age]])
    ans3=knn.predict([[preg , glu, blood, skin, insulin, bmi, pdf, age]])
    
    pos=0
    neg=0
    if(ans1==[[0]]):
        neg=neg+1
    else:
        pos=pos+1
    if(ans2==[[0]]):
        neg=neg+1
    else:
        pos=pos+1
    if(ans3==[[0]]):
        neg=neg+1
    else:
        pos=pos+1
        
    if(neg>pos):
        resultbox.configure(state='normal')
        resultbox.delete(0, END)
        resultbox.insert(END,"Diabetes Negative. You don't have diabetes")
        resultbox.configure(state='disabled')
    else:
        imp=kmeans.predict([[glu,insulin]])
        if(imp==[[0]]):
            resultbox.configure(state='normal')
            resultbox.delete(0, END)
            resultbox.insert(END,"Diabetes positive. Medical Assistance not required at moment.")
            resultbox.configure(state='disabled')
        else:
            resultbox.configure(state='normal')
            resultbox.delete(0, END)
            resultbox.insert(END,"Diabetes positive. Medical Assistance required as soon as possible.")
            resultbox.configure(state='disabled')


C = Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file = "bg1.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

label1=Label(root,text="  Diabetes Predictor", bg='#e3f4fc',font=("Helvetica", 36))
label1.grid(row=1,columnspan=2)

label2=Label(root,text="Enter the no. of pregnancies: ", bg='#e3f4fc')
label2.grid(row=3,column=0,sticky=E,padx=20,pady=5)

label3=Label(root,text="Enter the glucose level: ", bg='#e3f4fc')
label3.grid(row=4,column=0,sticky=E,padx=20,pady=5)

label4=Label(root,text="Enter the blood pressure: ", bg='#e3f4fc')
label4.grid(row=5,column=0,sticky=E,padx=20,pady=5)

label5=Label(root,text="Enter the skin thickness in mm: ", bg='#e3f4fc')
label5.grid(row=6,column=0,sticky=E,padx=20,pady=5)

label6=Label(root,text="Enter the insulin level: ", bg='#e3f4fc')
label6.grid(row=7,column=0,sticky=E,padx=20,pady=5)

label7=Label(root,text="Enter the BMI(Body Mass Index): ", bg='#e3f4fc')
label7.grid(row=8,column=0,sticky=E,padx=20,pady=5)

label8=Label(root,text="Enter the diabetes pedigree function: ", bg='#e3f4fc')
label8.grid(row=9,column=0,sticky=E,padx=20,pady=5)

label9=Label(root,text="Enter your age: ", bg='#e3f4fc')
label9.grid(row=10,column=0,sticky=E,padx=20,pady=5)

ip2=Entry(root,width=20)
ip2.grid(row=3,column=1,sticky=W)

ip3=Entry(root,width=20)
ip3.grid(row=4,column=1,sticky=W)

ip4=Entry(root,width=20)
ip4.grid(row=5,column=1,sticky=W)

ip5=Entry(root,width=20)
ip5.grid(row=6,column=1,sticky=W)

ip6=Entry(root,width=20)
ip6.grid(row=7,column=1,sticky=W)

ip7=Entry(root,width=20)
ip7.grid(row=8,column=1,sticky=W)

ip8=Entry(root,width=20)
ip8.grid(row=9,column=1,sticky=W)

ip9=Entry(root,width=20)
ip9.grid(row=10,column=1,sticky=W)

resultbox=Entry(root,width=60)
resultbox.grid(row=12,columnspan=2,sticky=E,padx=97,pady=7)
resultbox.configure(state='disabled')

button1=Button(root, text="Predict",command=action,width=10)
button1.grid(row=11,column=0,sticky=E, padx=100,pady=5)

button2=Button(root, text="Test Model",command=acc,width=10)
button2.grid(row=11,column=1,sticky=E,padx=100,pady=5)

a1=Entry(root,width=60)
a1.grid(row=15,columnspan=2,sticky=E,padx=97)
a1.configure(state='disabled')

a2=Entry(root,width=60)
a2.grid(row=16,columnspan=2,sticky=E,padx=97)
a2.configure(state='disabled')

a3=Entry(root,width=60)
a3.grid(row=17,columnspan=2,sticky=E,padx=97)
a3.configure(state='disabled')




C.grid()
root.mainloop()


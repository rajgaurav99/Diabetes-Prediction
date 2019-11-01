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

from tkinter import Button, PhotoImage, Entry, Label, E, W, Toplevel, Canvas, Tk, messagebox, END
root=Tk()
root.title("Diabetes Predictor")
root.geometry("600x500")
root.resizable(False,False)
root.configure(bg="#e3f4fc")




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
    preg=ip2.get()
    glu=ip3.get()
    blood=ip4.get()
    skin=ip5.get()
    insulin=ip6.get()
    bmi=ip7.get()
    pdf=ip8.get()
    age=ip9.get()

    if(preg.isdigit() and glu.isdigit() and blood.isdigit() and skin.isdigit() and insulin.isdigit() and bmi.replace('.', '', 1).isdigit() and pdf.replace('.', '', 1).isdigit() and age.isdigit()):
        ans1=clf.predict([[int(preg) , int(glu), int(blood), int(skin), int(insulin), float(bmi), float(pdf), int(age)]])
        ans2=gnb.predict([[int(preg) , int(glu), int(blood), int(skin), int(insulin), float(bmi), float(pdf), int(age)]])
        ans3=knn.predict([[int(preg) , int(glu), int(blood), int(skin), int(insulin), float(bmi), float(pdf), int(age)]])
        
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
            messagebox.showinfo("Result", "Diabetes Negative. You don't have diabetes")
        else:
            imp=kmeans.predict([[glu,insulin]])
            if(imp==[[0]]):
                resultbox.configure(state='normal')
                resultbox.delete(0, END)
                resultbox.insert(END,"Diabetes positive. Medical Assistance not required at moment.")
                resultbox.configure(state='disabled')
                messagebox.showinfo("Result", "Diabetes positive. Medical Assistance not required at moment.")
            else:
                resultbox.configure(state='normal')
                resultbox.delete(0, END)
                resultbox.insert(END,"Diabetes positive. Medical Assistance required as soon as possible.")
                resultbox.configure(state='disabled')
                messagebox.showinfo("Result", "Diabetes positive. Medical Assistance required as soon as possible.")
    else:
        messagebox.showinfo("Error", "Invalid Input or textbox left empty")



def clustering():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    newwin = Toplevel(root)
    newwin.resizable(False, False)
    newwin.title("Instructions for Usage")
    newwin.geometry("600x500")
    C = Canvas(newwin, bg="blue", height=250, width=300)
    filename = PhotoImage(file = "bg1.png")
    background_label = Label(newwin, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    l1=Label(newwin,text="Patient Clustering", bg='#e3f4fc',font=("Helvetica", 36))
    l1.grid(row=1,column=0,sticky=W,padx=120)
    
    label_desc=Label(newwin,text="For the purpose of Patient Clustering, the glucose and insulin levels are taken as the attributes to group the\n patients as they are only attributes that can be controlled by medical specialists.", bg='#e3f4fc')
    label_desc.grid(row=2,column=0,sticky=W,padx=10, pady=0)
    
    X=pd.read_csv('d2.csv')
    X=pd.DataFrame(X)
    
    kmeans = KMeans(n_clusters=2,init=np.array([[0,0],[189,846]]),n_init=1) 
    y_kmeans = kmeans.fit_predict(X)
    centres=kmeans.cluster_centers_
    centrex=[]
    centrey=[]
    for i in range(len(centres)):
        centrex.append(centres[i][0])
        centrey.append(centres[i][1])
    color=['red','green','blue','purple','black']

    
    figure1 = plt.Figure(figsize=(6,4), dpi=100)
    plot = figure1.add_subplot(1, 1, 1)
    for i in range(len(X)):
        plot.scatter(X.iloc[i,0],X.iloc[i,1],color=color[y_kmeans[i]])
    plot.scatter(centrex,centrey,color='black',s=200,label='Cluster Centroid')
    plot.legend()
    plot.set_xlabel('Glucose')
    plot.set_ylabel('Insulin')
    plot.set_title(' Clusters visualized from different colours')
    bar1 = FigureCanvasTkAgg(figure1, newwin)
    bar1.get_tk_widget().grid(row=3,column=0,sticky=W)
    C.grid()
    newwin.mainloop()


def instructions():
    newwin = Toplevel(root)
    newwin.resizable(False, False)
    newwin.title("Instructions for Usage")
    newwin.geometry("600x500")
    C = Canvas(newwin, bg="blue", height=250, width=300)
    filename = PhotoImage(file = "bg1.png")
    background_label = Label(newwin, image=filename)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    l1=Label(newwin,text="Instructions", bg='#e3f4fc',font=("Helvetica", 36))
    l1.grid(row=1,column=0,sticky=W,padx=170)
    
    label2key=Label(newwin,text="Blood Pressure: ", bg='#e3f4fc',font='Helvetica 12 bold')
    label2key.grid(row=2,column=0,sticky=W,padx=10, pady=10)
    label2ex=Label(newwin,text="Enter your blood pressure in mmHG. Value should be rounded to nearest integer.", bg='#e3f4fc')
    label2ex.grid(row=3,column=0,sticky=W,padx=10,pady=0)
    
    label3key=Label(newwin,text="Pregnancies: ", bg='#e3f4fc',font='Helvetica 12 bold')
    label3key.grid(row=4,column=0,sticky=W,padx=10, pady=10)
    label3ex=Label(newwin,text="Enter 0 if you are a male or female with no pregnancies. Else enter the no. of times you have been pregnant.", bg='#e3f4fc')
    label3ex.grid(row=5,column=0,sticky=W,padx=10,pady=0)
    
    label4key=Label(newwin,text="Glucose Level: ", bg='#e3f4fc',font='Helvetica 12 bold')
    label4key.grid(row=6,column=0,sticky=W,padx=10, pady=10)
    label4ex=Label(newwin,text="Glucose Level plays measures the amount of sugar present in your body. Enter the amount in mg/dl.", bg='#e3f4fc')
    label4ex.grid(row=7,column=0,sticky=W,padx=10,pady=0)
    
    label5key=Label(newwin,text="Insulin Level: ", bg='#e3f4fc',font='Helvetica 12 bold')
    label5key.grid(row=8,column=0,sticky=W,padx=10, pady=10)
    label5ex=Label(newwin,text="Insulin helps your body to absorb sugar. Should be entered in mIU/L", bg='#e3f4fc')
    label5ex.grid(row=9,column=0,sticky=W,padx=10,pady=0)
    
    label5key=Label(newwin,text="Body Mass Index (BMI): ", bg='#e3f4fc',font='Helvetica 12 bold')
    label5key.grid(row=8,column=0,sticky=W,padx=10, pady=10)
    label5ex=Label(newwin,text="BMI can be measured by dividing your weight by square of your height in metres.\n Enter exct float value upto 1 decimal place.", bg='#e3f4fc')
    label5ex.grid(row=9,column=0,sticky=W,padx=10,pady=0)
    
    label6key=Label(newwin,text="Diabetes Pedigree Function: ", bg='#e3f4fc',font='Helvetica 12 bold')
    label6key.grid(row=10,column=0,sticky=W,padx=10, pady=10)
    label6ex=Label(newwin,text="Gives diabetes estimate based on your family history. Value should be between 0 to 1\n having float values. Higher value means that there are more diabetic people\n in your family. Float value can be entered upto 3 decimal places.", bg='#e3f4fc')
    label6ex.grid(row=11,column=0,sticky=W,padx=10,pady=0)
    
    C.grid()
    newwin.mainloop()

C = Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file = "bg1.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

label1=Label(root,text="  Diabetes Predictor", bg='#e3f4fc',font=("Helvetica", 36))
label1.grid(row=1,columnspan=2)

button3=Button(root, text="View Instructions",command=instructions,width=20)
button3.grid(row=2,column=0)

button4=Button(root, text="View Patient Clustering",command=clustering,width=20)
button4.grid(row=2,column=1)

label2=Label(root,text="Enter the no. of pregnancies: ", bg='#e3f4fc')
label2.grid(row=3,column=0,sticky=E,padx=20,pady=2)

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


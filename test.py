# test1.py
#import the library
import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#Sidebar
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.title("🔗 Sources")
st.sidebar.info('[Given Data](https://drive.google.com/file/d/1cEMvten1WEJTae9xRw0JNezHrZZhhT9o/view?usp=sharing)'+'\r')
st.sidebar.info('[Source Code](https://drive.google.com/file/d/1cEMvten1WEJTae9xRw0JNezHrZZhhT9o/view?usp=sharing)'+'\r')
st.sidebar.title("🛈 About")
st.sidebar.info('Created and maintained by:'+'\r'+'[Eleni Giakoumi](eg.giakoumi@edu.cut.ac.cy)'+'[ Andreas Othonos](am.othonos@edu.cut.ac.cy)'+'[ Andriani Petrou](ae.petrou@edu.cut.ac.cy)')

st.image("1.png")
# add title
st.title('CEI 523 Assignment 2021 📈')
st.info('Case study for our given data for predictive maintenance')
st.text("")
st.markdown("### Given data")
st.markdown("---")
st.warning('Unclean Data')


# add dataset = pd.read_csv('https://github.com/itsheleng/test2/blob/3f6d2c0cfbdbeebbf2f6fb07fbdbee577004479c/data.csv', error_bad_lines=False )
st.sidebar.text("")
st.sidebar.text("")
#take the data
data=pd.read_csv("data.csv")
dataset=pd.read_csv("data.csv")
#print the data
st.write(data)
st.markdown("---")
st.success("Clean data")

#UDI,Product_ID,Type,Air_temperature,Process_temperature,Rotational_speed,Torque,Tool_wear,Machine_failure,TWF,HDF,PWF,OSF,RNF
y = data.Machine_failure
#we drop those columns beacasue some are odjects so we dont want them
dataset.drop(["Machine_failure","Product_ID","Type","UDI","TWF","HDF","PWF","OSF","RNF"],axis=1 , inplace=True)
X=dataset
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#Information for cleanning data
st.write(str("Checking for uniqueness in column Product_ID:   "),data['Product_ID'].is_unique)


st.write(str('Any missing data or NaN in the dataset:   '),data.isnull().values.any())


st.markdown("But we can see 0 values in **Tool wear** column")
st.text("Replace zero with mean number at column Tool wear")
m=data["Tool_wear"].mean()
data["Tool_wear"]=data["Tool_wear"].replace(0,m)
data["Tool_wear"]=data["Tool_wear"].astype("int")
st.write("  ")
data

st.write("  ")
st.markdown("### Algorithms we used for predictions (2)")

#LINEAR REGRESSION
st.markdown("   In the beggining we train our model using LinearSVC algorithm based on the Data modeling diagram.   We had sample of 10K labeled data that predicting a category.")
#linear
st.write("  ")
st.markdown("### LinearSVC")
train_model = LinearSVC(class_weight="balanced", dual=False, tol=1e-4, max_iter=1e5)
train_model.fit(X_train, y_train)
predictionsSVC = train_model.predict(X_test)
accurracy=accuracy_score(y_test, predictionsSVC)

st.markdown("Horizontal:Predicted, Vertical:Actual,0-NO,1-YES")

col1,col2=st.columns(2)
with col1:
 st.markdown("Confusion Matrix:")
 st.write(confusion_matrix(y_test, predictionsSVC))
with col2:
 st.markdown("Accuracy Score:")
 st.write(accuracy_score(y_test, predictionsSVC))
st.write(" ")
st.write("  ")
st.write("The Actual and Prediction Table of  LinearSVC")
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': predictionsSVC})
df1

st.write("  ")
#GAUSSIAN
st.markdown("### Gaussian Naive Bayes")
st.markdown("As we see the accuracy score of the **Linear SVC** was low,and we try the Gaussian Naive Bayes:")
gnb = GaussianNB()
yGNB_pred = gnb.fit(X_train, y_train).predict(X_test)
st.markdown("Horizontal:Predicted, Vertical:Actual,0-NO,1-YES")
predictionsGNB = gnb.predict(X_test)
col1,col2=st.columns(2)
with col1:
 st.markdown("Confusion Matrix:")
 st.write(confusion_matrix(y_test, predictionsGNB))
with col2:
 st.markdown("Accuracy Score:")
 st.write(accuracy_score(y_test, predictionsGNB))

st.write(" ")
st.write("The Actual and Prediction Table of Gaussian Naive Bayes")


# visualize regression line
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': predictionsGNB})
df2


# Visualise actual/predicted comparison
df2.plot(kind='bar',figsize=(1,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='red')
st.write(plt.show())

st.markdown("---")
st.error("### Conclusion")
st.markdown("Based on the algorithms we used we can see Gaussian Naive Bayes is better than LinearSVC algorithm, for our given data")

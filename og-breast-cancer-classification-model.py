# %% [markdown]
# # **Downloading Breast Cancer Dataset**

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:28.667823Z","iopub.execute_input":"2025-06-13T20:46:28.668111Z","iopub.status.idle":"2025-06-13T20:46:29.177271Z","shell.execute_reply.started":"2025-06-13T20:46:28.668090Z","shell.execute_reply":"2025-06-13T20:46:29.176341Z"}}
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

print("Path to dataset files:", path)

# %% [markdown]
# # **Import Modules for Analysis**

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:29.178817Z","iopub.execute_input":"2025-06-13T20:46:29.179093Z","iopub.status.idle":"2025-06-13T20:46:33.355501Z","shell.execute_reply.started":"2025-06-13T20:46:29.179071Z","shell.execute_reply":"2025-06-13T20:46:33.354629Z"}}
#Import modules for analysis

import numpy as np
import pandas as pd 
import math 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.spatial 
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # **Reading Breast Cencer Dataset**

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.356354Z","iopub.execute_input":"2025-06-13T20:46:33.356774Z","iopub.status.idle":"2025-06-13T20:46:33.384776Z","shell.execute_reply.started":"2025-06-13T20:46:33.356752Z","shell.execute_reply":"2025-06-13T20:46:33.383832Z"}}
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.385745Z","iopub.execute_input":"2025-06-13T20:46:33.386040Z","iopub.status.idle":"2025-06-13T20:46:33.391032Z","shell.execute_reply.started":"2025-06-13T20:46:33.386009Z","shell.execute_reply":"2025-06-13T20:46:33.389961Z"}}
#Printing Dataset Dimensions
print(data.shape)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.393218Z","iopub.execute_input":"2025-06-13T20:46:33.393941Z","iopub.status.idle":"2025-06-13T20:46:33.450124Z","shell.execute_reply.started":"2025-06-13T20:46:33.393918Z","shell.execute_reply":"2025-06-13T20:46:33.449321Z"}}
#Print the first 5 rows of the dataset
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.451176Z","iopub.execute_input":"2025-06-13T20:46:33.451518Z","iopub.status.idle":"2025-06-13T20:46:33.457469Z","shell.execute_reply.started":"2025-06-13T20:46:33.451490Z","shell.execute_reply":"2025-06-13T20:46:33.456573Z"}}
#Inspect data types
print(data.dtypes)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.458305Z","iopub.execute_input":"2025-06-13T20:46:33.458558Z","iopub.status.idle":"2025-06-13T20:46:33.480404Z","shell.execute_reply.started":"2025-06-13T20:46:33.458539Z","shell.execute_reply":"2025-06-13T20:46:33.479118Z"}}
#Check for missing values
print(data.isnull().sum())

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.481342Z","iopub.execute_input":"2025-06-13T20:46:33.481641Z","iopub.status.idle":"2025-06-13T20:46:33.504553Z","shell.execute_reply.started":"2025-06-13T20:46:33.481617Z","shell.execute_reply":"2025-06-13T20:46:33.503602Z"}}
#Types of Cancer: Malignant and Benign
print(data["diagnosis"].unique())

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.505741Z","iopub.execute_input":"2025-06-13T20:46:33.506074Z","iopub.status.idle":"2025-06-13T20:46:33.528578Z","shell.execute_reply.started":"2025-06-13T20:46:33.506052Z","shell.execute_reply":"2025-06-13T20:46:33.527590Z"}}
# Calculating the proportions of cases we have in each cancer type
print(data["diagnosis"].value_counts()/data.shape[0])

# %% [markdown]
# # **Cleaning Data**

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.529498Z","iopub.execute_input":"2025-06-13T20:46:33.529757Z","iopub.status.idle":"2025-06-13T20:46:33.566618Z","shell.execute_reply.started":"2025-06-13T20:46:33.529739Z","shell.execute_reply":"2025-06-13T20:46:33.565589Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.567613Z","iopub.execute_input":"2025-06-13T20:46:33.567875Z","iopub.status.idle":"2025-06-13T20:46:33.591881Z","shell.execute_reply.started":"2025-06-13T20:46:33.567855Z","shell.execute_reply":"2025-06-13T20:46:33.590773Z"}}
#Dropping the ID and Unnamed: 32 columns as they are not needed for analysis

data.drop (['id', 'Unnamed: 32'], axis = 1, inplace = True)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.592974Z","iopub.execute_input":"2025-06-13T20:46:33.593288Z","iopub.status.idle":"2025-06-13T20:46:33.632228Z","shell.execute_reply.started":"2025-06-13T20:46:33.593260Z","shell.execute_reply":"2025-06-13T20:46:33.631205Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.633642Z","iopub.execute_input":"2025-06-13T20:46:33.633883Z","iopub.status.idle":"2025-06-13T20:46:33.651317Z","shell.execute_reply.started":"2025-06-13T20:46:33.633865Z","shell.execute_reply":"2025-06-13T20:46:33.650262Z"}}
#Encoding the diagnosis column to 0 and 1 where 0 is Benign and 1 is Malignant
if data.diagnosis[0]=="M" or data.diagnosis[0]=="B":
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.654189Z","iopub.execute_input":"2025-06-13T20:46:33.654500Z","iopub.status.idle":"2025-06-13T20:46:33.692064Z","shell.execute_reply.started":"2025-06-13T20:46:33.654480Z","shell.execute_reply":"2025-06-13T20:46:33.691071Z"}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:33.693361Z","iopub.execute_input":"2025-06-13T20:46:33.693852Z","iopub.status.idle":"2025-06-13T20:46:34.120393Z","shell.execute_reply.started":"2025-06-13T20:46:33.693823Z","shell.execute_reply":"2025-06-13T20:46:34.119347Z"}}
#Plotting the distribution of the diagnosis column to see the proportion of Malignant and Benign cases
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=data)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:34.121691Z","iopub.execute_input":"2025-06-13T20:46:34.122027Z","iopub.status.idle":"2025-06-13T20:46:34.127744Z","shell.execute_reply.started":"2025-06-13T20:46:34.121993Z","shell.execute_reply":"2025-06-13T20:46:34.126712Z"}}
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# %% [markdown]
# # **Normalising the Data**

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:34.129089Z","iopub.execute_input":"2025-06-13T20:46:34.129406Z","iopub.status.idle":"2025-06-13T20:46:34.161428Z","shell.execute_reply.started":"2025-06-13T20:46:34.129356Z","shell.execute_reply":"2025-06-13T20:46:34.160100Z"}}
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:34.162516Z","iopub.execute_input":"2025-06-13T20:46:34.162788Z","iopub.status.idle":"2025-06-13T20:46:34.225618Z","shell.execute_reply.started":"2025-06-13T20:46:34.162768Z","shell.execute_reply":"2025-06-13T20:46:34.224748Z"}}
data.describe().T

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:46:34.227038Z","iopub.execute_input":"2025-06-13T20:46:34.227329Z","iopub.status.idle":"2025-06-13T20:46:36.814094Z","shell.execute_reply.started":"2025-06-13T20:46:34.227309Z","shell.execute_reply":"2025-06-13T20:46:36.813111Z"}}
#Import plotting modules
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout 
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:59:11.826450Z","iopub.execute_input":"2025-06-13T20:59:11.826770Z","iopub.status.idle":"2025-06-13T20:59:11.882253Z","shell.execute_reply.started":"2025-06-13T20:59:11.826747Z","shell.execute_reply":"2025-06-13T20:59:11.881424Z"}}
p = data.describe().T 
p = p.round(4) 
table = go.Table(
    columnwidth=[0.8]+[0.5]*8, 
    header=dict(
        values=['Attribute'] + list(p.columns), line = dict(color='#506784'),
        fill = dict(color='lightblue'),
), 
    cells=dict(
        values=[p.index] + [p[k].tolist() for k in p.columns[:]], line = dict(color='#506784'),
        fill = dict(color=['rgb(173, 216, 220)', '#f5f5fa'])
) )

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:59:13.965189Z","iopub.execute_input":"2025-06-13T20:59:13.965545Z","iopub.status.idle":"2025-06-13T20:59:14.143114Z","shell.execute_reply.started":"2025-06-13T20:59:13.965508Z","shell.execute_reply":"2025-06-13T20:59:14.142174Z"}}
B, M = data['diagnosis'].value_counts()      # unpack the two counts
s = [B, M]

print(data['diagnosis'].value_counts())      # use the same variable name

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    plt.bar([0, 1], s, align='center', label='Count')
    plt.xticks([0, 1], ['Benign', 'Malignant'])  # optional: nicer x-labels
    plt.ylabel('')
    plt.xlabel('')
    plt.legend(loc='best')
    plt.tight_layout()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:59:24.745764Z","iopub.execute_input":"2025-06-13T20:59:24.746144Z","iopub.status.idle":"2025-06-13T20:59:24.753536Z","shell.execute_reply.started":"2025-06-13T20:59:24.746119Z","shell.execute_reply":"2025-06-13T20:59:24.752471Z"}}
B, M = data['diagnosis'].value_counts()
trace1 = go.Bar(y = (M, B), x = ['malignant', 'benign'],opacity = 0.8)
trace2 = go.Pie(labels = ['Benign','Malignant'], values = data['diagnosis'].value_counts(), textfont=dict(size=15), opacity = 0.8)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T20:59:57.038020Z","iopub.execute_input":"2025-06-13T20:59:57.038785Z","iopub.status.idle":"2025-06-13T21:00:01.419155Z","shell.execute_reply.started":"2025-06-13T20:59:57.038755Z","shell.execute_reply":"2025-06-13T21:00:01.418074Z"}}
mean_col = [col for col in data.columns if col.endswith('_mean')] 
for i in range(len(mean_col)):
    sns.FacetGrid(data,hue="diagnosis",aspect=3,margin_titles=True).map(sns.kdeplot,mean_col[i],shade= True).add_legend() 

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:00:22.203643Z","iopub.execute_input":"2025-06-13T21:00:22.204014Z","iopub.status.idle":"2025-06-13T21:00:24.432356Z","shell.execute_reply.started":"2025-06-13T21:00:22.203992Z","shell.execute_reply":"2025-06-13T21:00:24.431291Z"}}
#Correlation Map
f,ax = plt.subplots(figsize=(18, 18))
cmap = sns.diverging_palette( 240 , 10 , as_cmap = True ) 
sns.heatmap(data.corr(), cmap='Blues',annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.xticks(fontsize=11,rotation=70)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:00:37.558162Z","iopub.execute_input":"2025-06-13T21:00:37.558516Z","iopub.status.idle":"2025-06-13T21:01:06.684987Z","shell.execute_reply.started":"2025-06-13T21:00:37.558490Z","shell.execute_reply":"2025-06-13T21:01:06.683970Z"}}
from pylab import rcParams 
rcParams['figure.figsize'] = 8,5

cols = ['radius_mean', 'texture_mean', 'perimeter_mean',
'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean','diagnosis']
sns_plot = sns.pairplot(data=data[cols],hue='diagnosis')

# %% [markdown]
# # **Splitting the data into train and test sets**

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:02:14.516163Z","iopub.execute_input":"2025-06-13T21:02:14.516523Z","iopub.status.idle":"2025-06-13T21:02:14.526519Z","shell.execute_reply.started":"2025-06-13T21:02:14.516496Z","shell.execute_reply":"2025-06-13T21:02:14.525269Z"}}
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42) #30% of data is kept for testing

# Shuffle the dataset 
# shuffle_df = shuffle(data)
# np.random.shuffle(shuffle_df.values)

shuffle_df = data.sample(frac=1)

# Define a size for the train set 
train_size = int(0.8 * len(data))

# Splitting the dataset 
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

y_train = train_set.diagnosis.values
X_train = train_set.drop(["diagnosis"], axis=1)
y_test  = test_set.diagnosis.values
X_test  = test_set.drop(["diagnosis"], axis=1)

# %% [markdown]
# # **Building K-Nearest Neighbours for Classification without Sklearn Modules**

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:02:48.140458Z","iopub.execute_input":"2025-06-13T21:02:48.140776Z","iopub.status.idle":"2025-06-13T21:02:48.149163Z","shell.execute_reply.started":"2025-06-13T21:02:48.140754Z","shell.execute_reply":"2025-06-13T21:02:48.148103Z"}}
class KNN:
    def __init__(self, k):
        self.k = k
        
    #Fit function to keep the data with itself, since KNN does not perform any explicit training process.   
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        
     #It calculate the Euclidean distance and returns how similar two examples are  
    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)
        
        
    # In predict function, it predicts the class for testing instance using the complete training data.
    #  1- First it calculates the distance between a test data point and every training data point,
    #  2- It sorts the distances and picks K nearest distances(first K entries) from it.
    #  3- Gets the labels of the selected K neighbors. The most common label(label with a majority vote) will be the
    #     predicted label for our test data point.
    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(self.X_train)):
                dist = scipy.spatial.distance.euclidean(self.X_train[j] , X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(self.y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output
    
    # It calculate the score for our model based on the test data
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:02:58.380947Z","iopub.execute_input":"2025-06-13T21:02:58.381280Z","iopub.status.idle":"2025-06-13T21:02:58.386348Z","shell.execute_reply.started":"2025-06-13T21:02:58.381255Z","shell.execute_reply":"2025-06-13T21:02:58.385220Z"}}
#Store neighbours and errors in an empty list
neighbours = []
errors = []

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:03:06.367020Z","iopub.execute_input":"2025-06-13T21:03:06.367334Z","iopub.status.idle":"2025-06-13T21:03:12.505413Z","shell.execute_reply.started":"2025-06-13T21:03:06.367313Z","shell.execute_reply":"2025-06-13T21:03:12.504545Z"}}
for k in range(1,30, 2):
    neighbours.append(k)
    clf = KNN(k)
    clf.fit(X_train.to_numpy(), y_train)
    score = clf.score(X_test.to_numpy(), y_test)
    errors.append(1-score)
    print("Number of neighbours : ",k,", Accuracy = ",score)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:03:24.447567Z","iopub.execute_input":"2025-06-13T21:03:24.447918Z","iopub.status.idle":"2025-06-13T21:03:24.860112Z","shell.execute_reply.started":"2025-06-13T21:03:24.447896Z","shell.execute_reply":"2025-06-13T21:03:24.859348Z"}}
clf = KNN(9)
clf.fit(X_train.to_numpy(), y_train)
score = clf.score(X_test.to_numpy(), y_test)
print("Accuracy of KNN = ",score)

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:03:36.252702Z","iopub.execute_input":"2025-06-13T21:03:36.252996Z","iopub.status.idle":"2025-06-13T21:03:36.259000Z","shell.execute_reply.started":"2025-06-13T21:03:36.252977Z","shell.execute_reply":"2025-06-13T21:03:36.257961Z"}}
# Choosing the value of K which gave the least error
MSE = [x for x in errors]
optimal_k = neighbours[MSE.index(min(MSE))]
print("Optimal K value is: "+str(optimal_k))
print("Accuracy at K="+str(optimal_k)+" is: "+str(1-float(MSE[optimal_k])))

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:03:46.132929Z","iopub.execute_input":"2025-06-13T21:03:46.133245Z","iopub.status.idle":"2025-06-13T21:03:46.310957Z","shell.execute_reply.started":"2025-06-13T21:03:46.133223Z","shell.execute_reply":"2025-06-13T21:03:46.309993Z"}}
#Plotting the error values against K values
plt.figure(figsize=(10, 6))
plt.plot(neighbours, MSE)
plt.xlabel('K value --->')
plt.ylabel('Error  --->')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:12:12.686788Z","iopub.execute_input":"2025-06-13T21:12:12.687145Z","iopub.status.idle":"2025-06-13T21:12:16.585799Z","shell.execute_reply.started":"2025-06-13T21:12:12.687115Z","shell.execute_reply":"2025-06-13T21:12:16.584747Z"}}
!pip install streamlit

# %% [code] {"execution":{"iopub.status.busy":"2025-06-13T21:12:33.382566Z","iopub.execute_input":"2025-06-13T21:12:33.382916Z","iopub.status.idle":"2025-06-13T21:12:33.391360Z","shell.execute_reply.started":"2025-06-13T21:12:33.382887Z","shell.execute_reply":"2025-06-13T21:12:33.390353Z"}}
%%writefile app.py
import streamlit as st, joblib, numpy as np
from pathlib import Path

# ---------- constants ----------
MODEL_PATH = Path("breast_cancer_clf.pkl")   # put your model here
TEST_ACCURACY = 0.971                        # update with your own score

# ---------- load model ----------
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        st.error(f"❌ Model file not found: {path.resolve()}")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ---------- UI ----------
st.set_page_config(page_title="Breast-Tumor Classifier", page_icon="🩺")
st.title("Breast-tumor classifier")
st.caption(f"Hold-out test accuracy: **{TEST_ACCURACY:.1%}**")

st.markdown("Enter feature measurements below (values from the Wisconsin Diagnostic dataset).")

# For brevity only a few inputs are shown—repeat for all features you want:
radius_mean  = st.number_input("Mean radius (mm)",  0.0,  50.0, step=0.01)
texture_mean = st.number_input("Mean texture",      0.0, 100.0, step=0.01)
perimeter_mean = st.number_input("Mean perimeter",  0.0, 300.0, step=0.01)
area_mean    = st.number_input("Mean area",         0.0, 2500.0, step=1.0)

if st.button("Classify"):
    # assemble feature vector in the order the model expects
    X = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
    prob_malignant = model.predict_proba(X)[0, 1]
    label = "Malignant" if prob_malignant >= 0.50 else "Benign"

    st.success(f"**{label}**  (Probability malignant: {prob_malignant:.1%})")
    st.caption("_For research use only – not a diagnostic device._")
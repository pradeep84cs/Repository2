import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


st.title("Advanced AI/ML Comparison System")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

columns = [
"Pregnancies","Glucose","BloodPressure","SkinThickness",
"Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]

data = pd.read_csv(url,names=columns)

st.subheader("Dataset Preview")
st.write(data.head())

X = data.drop("Outcome",axis=1)
y = data["Outcome"]

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Define Models
# --------------------------------------------------

models = {

"Logistic Regression": LogisticRegression(max_iter=1000),

"KNN": KNeighborsClassifier(5),

"Decision Tree": DecisionTreeClassifier(),

"Random Forest": RandomForestClassifier(100),

"SVM": SVC(probability=True),

"Neural Network": MLPClassifier(max_iter=1000),

"XGBoost": XGBClassifier(eval_metric="logloss"),

"LightGBM": LGBMClassifier(),

"CatBoost": CatBoostClassifier(verbose=0),

"TabNet": TabNetClassifier()
}

# --------------------------------------------------
# Function to Evaluate Model
# --------------------------------------------------

def evaluate_model(model,name):

    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    if hasattr(model,"predict_proba"):
        prob=model.predict_proba(X_test)[:,1]
    else:
        prob=pred

    acc=accuracy_score(y_test,pred)
    prec=precision_score(y_test,pred)
    rec=recall_score(y_test,pred)
    f1=f1_score(y_test,pred)
    auc=roc_auc_score(y_test,prob)

    cv=cross_val_score(model,X_train,y_train,cv=5).mean()

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)
    st.write("ROC-AUC:",auc)
    st.write("Cross Validation:",cv)

    # Confusion Matrix
    cm=confusion_matrix(y_test,pred)

    fig,ax=plt.subplots()

    sns.heatmap(cm,annot=True,ax=ax)

    ax.set_title(name+" Confusion Matrix")

    st.pyplot(fig)

    # ROC Curve
    fpr,tpr,_=roc_curve(y_test,prob)

    fig2,ax2=plt.subplots()

    ax2.plot(fpr,tpr)

    ax2.plot([0,1],[0,1],'--')

    ax2.set_title(name+" ROC Curve")

    st.pyplot(fig2)

    # Precision Recall Curve
    precision,recall,_=precision_recall_curve(y_test,prob)

    fig3,ax3=plt.subplots()

    ax3.plot(recall,precision)

    ax3.set_title(name+" Precision Recall")

    st.pyplot(fig3)

    return acc,prec,rec,f1,auc,cv,pred,prob


# --------------------------------------------------
# Run Individual Algorithms
# --------------------------------------------------

st.subheader("Run Individual Algorithms")

for name,model in models.items():

    if st.button("Run "+name):

        evaluate_model(model,name)

# --------------------------------------------------
# Compare All Algorithms
# --------------------------------------------------

st.subheader("Compare All Algorithms")

if st.button("Run Full Comparison"):

    results=[]
    roc_data={}
    pr_data={}
    predictions={}

    for name,model in models.items():

        acc,prec,rec,f1,auc,cv,pred,prob = evaluate_model(model,name)

        results.append([name,acc,prec,rec,f1,auc,cv])

        fpr,tpr,_=roc_curve(y_test,prob)
        roc_data[name]=(fpr,tpr)

        precision,recall,_=precision_recall_curve(y_test,prob)
        pr_data[name]=(precision,recall)

        predictions[name]=pred

    df=pd.DataFrame(results,columns=[
        "Algorithm","Accuracy","Precision","Recall","F1","ROC-AUC","Cross Validation"
    ])

    st.subheader("Performance Table")

    st.write(df)

# --------------------------------------------------
# Algorithm Accuracy Comparison Chart
# --------------------------------------------------

    fig,ax=plt.subplots()

    ax.bar(df["Algorithm"],df["Accuracy"])

    plt.xticks(rotation=45)

    ax.set_title("Algorithm Accuracy Comparison Chart")

    st.pyplot(fig)

# --------------------------------------------------
# Model Ranking Chart
# --------------------------------------------------

    df_sorted=df.sort_values("Accuracy",ascending=False)

    fig2,ax2=plt.subplots()

    ax2.barh(df_sorted["Algorithm"],df_sorted["Accuracy"])

    ax2.set_title("Model Ranking Chart")

    st.pyplot(fig2)

# --------------------------------------------------
# ROC Curve Comparison
# --------------------------------------------------

    fig3,ax3=plt.subplots()

    for name,(fpr,tpr) in roc_data.items():

        ax3.plot(fpr,tpr,label=name)

    ax3.plot([0,1],[0,1],'--')

    ax3.legend()

    ax3.set_title("ROC Curve Comparison")

    st.pyplot(fig3)

# --------------------------------------------------
# Precision Recall Curve
# --------------------------------------------------

    fig4,ax4=plt.subplots()

    for name,(precision,recall) in pr_data.items():

        ax4.plot(recall,precision,label=name)

    ax4.legend()

    ax4.set_title("Precision Recall Curve")

    st.pyplot(fig4)

# --------------------------------------------------
# Confusion Matrices
# --------------------------------------------------

    st.subheader("Confusion Matrices")

    for name,pred in predictions.items():

        fig_cm,ax_cm=plt.subplots()

        sns.heatmap(confusion_matrix(y_test,pred),annot=True,ax=ax_cm)

        ax_cm.set_title(name+" Confusion Matrix")

        st.pyplot(fig_cm)

# --------------------------------------------------
# SHAP Explainable AI
# --------------------------------------------------

    st.subheader("SHAP Explainable AI")

    model=RandomForestClassifier()

    model.fit(X_train,y_train)

    explainer=shap.TreeExplainer(model)

    shap_values=explainer.shap_values(X_test)

    fig_shap=plt.figure()

    shap.summary_plot(shap_values,X_test,show=False)

    st.pyplot(fig_shap)

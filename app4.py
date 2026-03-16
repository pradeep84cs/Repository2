import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


st.title("Advanced Diabetes Prediction Research System")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

columns = [
"Pregnancies","Glucose","BloodPressure","SkinThickness",
"Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]

data = pd.read_csv(url,names=columns)

st.subheader("Dataset Preview")
st.write(data.head())

st.write("Total Records:",data.shape[0])

# ----------------------------------------------------
# Feature Correlation Heatmap
# ----------------------------------------------------

st.subheader("Feature Correlation Heatmap")

fig_corr, ax_corr = plt.subplots()

sns.heatmap(data.corr(),annot=True,cmap="coolwarm",ax=ax_corr)

st.pyplot(fig_corr)

# ----------------------------------------------------
# Feature / Target
# ----------------------------------------------------

X = data.drop("Outcome",axis=1)
y = data["Outcome"]

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------
# Define Models + Hyperparameter Grids
# ----------------------------------------------------

models = {

"Logistic Regression":(
LogisticRegression(max_iter=1000),
{"C":[0.01,0.1,1,10]}
),

"KNN":(
KNeighborsClassifier(),
{"n_neighbors":[3,5,7,9]}
),

"Decision Tree":(
DecisionTreeClassifier(),
{"max_depth":[3,5,7,None]}
),

"Random Forest":(
RandomForestClassifier(),
{"n_estimators":[50,100],
 "max_depth":[5,10,None]}
),

"SVM":(
SVC(probability=True),
{"C":[0.1,1,10],
 "kernel":["linear","rbf"]}
),

"Neural Network":(
MLPClassifier(max_iter=1000),
{"hidden_layer_sizes":[(10,),(20,),(30,)],
 "activation":["relu","tanh"]}
)

}

results=[]
roc_data={}
pr_data={}
predictions={}
best_models={}

st.subheader("Run Full Model Training")

if st.button("Train and Evaluate All Models"):

    for name,(model,params) in models.items():

        st.write("Training:",name)

        grid=GridSearchCV(model,params,cv=5)

        grid.fit(X_train,y_train)

        best=grid.best_estimator_

        best_models[name]=best

        pred=best.predict(X_test)

        prob=best.predict_proba(X_test)[:,1]

        acc=accuracy_score(y_test,pred)
        prec=precision_score(y_test,pred)
        rec=recall_score(y_test,pred)
        f1=f1_score(y_test,pred)
        auc=roc_auc_score(y_test,prob)

        # Cross Validation
        cv_score=cross_val_score(best,X_train,y_train,cv=5).mean()

        results.append([name,acc,prec,rec,f1,auc,cv_score])

        predictions[name]=pred

        fpr,tpr,_=roc_curve(y_test,prob)
        roc_data[name]=(fpr,tpr)

        precision,recall,_=precision_recall_curve(y_test,prob)
        pr_data[name]=(precision,recall)

    df=pd.DataFrame(results,columns=[
    "Algorithm","Accuracy","Precision","Recall",
    "F1","ROC-AUC","Cross Validation"
    ])

    st.subheader("Model Performance Table")

    st.write(df)

# ----------------------------------------------------
# Accuracy Comparison Chart
# ----------------------------------------------------

    fig_acc,ax_acc=plt.subplots()

    ax_acc.bar(df["Algorithm"],df["Accuracy"])

    ax_acc.set_xlabel("Algorithm")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Algorithm Accuracy Comparison")

    plt.xticks(rotation=45)

    st.pyplot(fig_acc)

# ----------------------------------------------------
# Model Ranking Chart
# ----------------------------------------------------

    df_sorted=df.sort_values("Accuracy",ascending=False)

    fig_rank,ax_rank=plt.subplots()

    ax_rank.barh(df_sorted["Algorithm"],df_sorted["Accuracy"])

    ax_rank.set_xlabel("Accuracy")
    ax_rank.set_ylabel("Algorithm")
    ax_rank.set_title("Model Ranking Chart")

    st.pyplot(fig_rank)

# ----------------------------------------------------
# ROC Curve Comparison
# ----------------------------------------------------

    fig_roc,ax_roc=plt.subplots()

    for name,(fpr,tpr) in roc_data.items():

        ax_roc.plot(fpr,tpr,label=name)

    ax_roc.plot([0,1],[0,1],'--')

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve Comparison")
    ax_roc.legend()

    st.pyplot(fig_roc)

# ----------------------------------------------------
# Precision Recall Curve
# ----------------------------------------------------

    fig_pr,ax_pr=plt.subplots()

    for name,(precision,recall) in pr_data.items():

        ax_pr.plot(recall,precision,label=name)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision Recall Curve")
    ax_pr.legend()

    st.pyplot(fig_pr)

# ----------------------------------------------------
# Confusion Matrices
# ----------------------------------------------------

    st.subheader("Confusion Matrices")

    for name,pred in predictions.items():

        fig_cm,ax_cm=plt.subplots()

        sns.heatmap(confusion_matrix(y_test,pred),
        annot=True,ax=ax_cm)

        ax_cm.set_title(name+" Confusion Matrix")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        st.pyplot(fig_cm)

# ----------------------------------------------------
# SHAP Explainable AI
# ----------------------------------------------------

    st.subheader("SHAP Explainable AI")

    model=list(best_models.values())[0]

    explainer=shap.Explainer(model,X_train)

    shap_values=explainer(X_test)

    fig_shap=plt.figure()

    shap.summary_plot(shap_values,X_test,show=False)

    st.pyplot(fig_shap)

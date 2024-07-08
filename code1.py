import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc as sklearn_auc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.simplefilter(action="ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 170)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df = pd.read_csv("https://github.com/Ganga-pavani/tasks/blob/main/Churn-Data.csv")
df.head()
df.info()
df.shape
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x =="Yes" else 0 )
df.head()
df.isnull().sum()
df.dropna(inplace = True)
df.isnull().sum()
df.shape
def grab_col_names(dataframe, cat_th=10, car_th=20):
    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O" and col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")
    print(f"cat_cols : {cat_cols}")
    print(f"num_cols : {num_cols}")
    print(f"cat_but_car : {cat_but_car}")
    
    return cat_cols, num_cols, cat_but_car

grab_col_names(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print("#############")
print(f"Cat_Cols : {cat_cols}")
print("#############")
print(f"Num_Cols : {num_cols}")
print("#############")
print(f"Cat_But_Car : {cat_but_car}")
def cat_summary(dataframe , col_name , plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts()/len(df)}))
    print("##########################################")
    if plot:
        fig, axs = plt.subplots(1,2 , figsize=(12,6))
        sns.countplot(x=col_name, data=dataframe, ax=axs[0])
        axs[0].set_title("Frequency of" + col_name)
        axs[0].tick_params(axis='x', rotation=90)
        values= dataframe[col_name].value_counts()
        axs[1].pie(values, labels=values.index, autopct='%1.1f%%')
        axs[1].set_title("Proportion of " + col_name)
        plt.show()

for col in cat_cols:
    cat_summary(df,col, plot =True)
def num_summary(dataframe, col_name , plot=False):
    print(dataframe[col_name].describe())
    print("##########################################")
    if plot:
        plt.figure(figsize=(8,4))
        sns.histplot(dataframe[col_name], kde=True)
        plt.title("Distribution of" + col_name)
        plt.show()
    
for col in num_cols:
    num_summary(df, col, plot=True)
def num_summary(dataframe, col_name , plot=False):
    print(dataframe[col_name].describe())
    print("##########################################")
    if plot:
        plt.figure(figsize=(8,4))
        sns.histplot(dataframe[col_name], kde=True)
        plt.title("Distribution of" + col_name)
        plt.show()
    
for col in num_cols:
    num_summary(df, col, plot=True)
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.countplot(x=col_name, data=dataframe, ax=axs[0])
        axs[0].set_title("Frequency of " + col_name)
        axs[0].tick_params(axis='x', rotation=90)
        values = dataframe[col_name].value_counts()
        axs[1].pie(values, labels=values.index, autopct='%1.1f%%')
        axs[1].set_title("Proportion of " + col_name)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=False)
    
def cat_summary_with_target(dataframe, col_name, target, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    cross_tab = pd.crosstab(dataframe[col_name], dataframe[target], normalize='index') * 100
    print(cross_tab)
    print("##########################################")
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.countplot(x=col_name, hue=target, data=dataframe, ax=axs[0])
        axs[0].set_title(f'Frequency of {col_name} by {target}')
        axs[0].tick_params(axis='x', rotation=90)
        cross_tab.plot(kind='bar', stacked=True, ax=axs[1])
        axs[1].set_title(f'Proportion of {col_name} by {target}')
        axs[1].legend(title=target)
        plt.show()

for col in cat_cols:
    cat_summary_with_target(df, col, 'Churn', plot=True)

grouped = df.groupby('Churn')[num_cols].mean()
print(grouped)

for col in num_cols:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=grouped.index, y=grouped[col],width=0.4)
    plt.title(f'Average of {col} by Churn')
    plt.ylabel('Average Value')
    plt.xlabel('Churn')
    plt.xticks(ticks=[0, 1], labels=['Not Churned (0)', 'Churned (1)'])
    plt.show()
df[num_cols].corr()
f,ax = plt.subplots(figsize=[15,10])
sns.heatmap(df[num_cols].corr(), annot = True, fmt=".2f" ,ax=ax , cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    IQR = quantile3 - quantile1
    low_limit = quantile1 - 1.5 * IQR
    up_limit = quantile3 + 1.5 * IQR
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit) ].any(axis=None):
        return True
    else:
        return False
    
for col in num_cols:
    print(col, outlier_thresholds(df, col))
for col in num_cols:
    print(col, check_outlier(df, col))
print(df[["tenure","MonthlyCharges","TotalCharges"]].describe().T)
def missing_values_table(dataframe, na_name=True):
    na_columns = [col for col in dataframe if dataframe[col].isnull().sum()>0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns
missing_columns = missing_values_table(df)
print("Columns with missing values:", missing_columns)

missing_values_table(df, True)
na_cols = missing_values_table(df, True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalChargesPerMonth'] = df['TotalCharges'] / df['tenure'].replace(0, np.nan)
df['TotalChargesPerMonth'].fillna(0, inplace=True)
bins = [0, 12, 24, 36, 48, 60, np.inf]
labels = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '60+ months']
df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)

service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumberOfServicesUsed'] = df[service_columns].apply(lambda x: sum(x == 'Yes'), axis=1)
df['MonthlyChargesPerService'] = df['MonthlyCharges'] / df['NumberOfServicesUsed'].replace(0, np.nan)
df['MonthlyChargesPerService'].fillna(0, inplace=True)
df['SeniorCitizenBinary'] = df['SeniorCitizen'].apply(lambda x: 'Yes' if x == 1 else 'No')
df['ContractType'] = df['Contract'].apply(lambda x: 'Monthly' if x == 'Month-to-month' else 'Long-term')
df['StreamingServicesUsed'] = df[['StreamingTV', 'StreamingMovies']].apply(lambda x: 'Yes' if 'Yes' in x.values else 'No', axis=1)
print(df[['TotalChargesPerMonth', 'TenureGroup', 'NumberOfServicesUsed', 'MonthlyChargesPerService', 'SeniorCitizenBinary', 'ContractType', 'StreamingServicesUsed']].head())
df.head()
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
print(df.head())
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype == "O"
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)
print(df.head())
X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier( verbose=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {round(accuracy, 2)}")
print(f"Recall: {round(recall, 2)}")
print(f"Precision: {round(precision, 2)}")
print(f"F1: {round(f1, 2)}")
print(f"AUC: {round(roc_auc, 2)}")
def plot_importance(model, features, save=False):
    feature_importances = model.get_feature_importance()
    feature_imp = pd.DataFrame({"Value": feature_importances, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set_theme(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title("Features Importance")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
plot_importance(model, X)  
def plot_roc_auc(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc_value = sklearn_auc(fpr, tpr)  
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve for CatBoost Model')
    plt.legend(loc="lower right")
    plt.show()


plot_roc_auc(model, X_test, y_test)




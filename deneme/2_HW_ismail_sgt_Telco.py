import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

####################  Görev 1 : Keşifçi Veri Analizi   ###################################
####################                                   ###################################

df = pd.read_csv("3_Feature_Engineering/datasets/TelcoCustomerChurn-230423-212029.csv")

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

df.dtypes

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

#  TotalCharges   tipi "object" geldi. bunun float64'e (INTEGER) dönüştürülmesi gerekiyor.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dtypes


# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

num_cols = [col for col in num_cols if col not in "customerID"]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

# Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# tenure False
# MonthlyCharges False
# TotalCharges False

# aykırı değer yoktur



# Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)


# TotalCharges  değişkeninde 11 adet eksik gözlem vardır.

############################  Görev 2 : Feature Engineering  #####################################
############################                                 #####################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

# TotalCharges değişkeninde 11 adet boşluk var. cinsiyet dağılına göre ortalamaya bakalım ve buna göre boş değerleri
# dolduralım.

# VEYA

# toplam 11 adet boş değer olduğu için drop edilebilir. toplam gözlem sayısına göre çok küçük bir oran.

df.groupby("gender").agg({"TotalCharges": ["mean", "median"]})
df["TotalCharges"].describe()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# boş değerler median ile dolduruldu.

df.isnull().sum()
# .......................................................


# Adım 2: Yeni değişkenler oluşturunuz


df["tenure"].describe() # min:0 ve max:72 ay
df.head()

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE"] = "1-year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE"] = "2-year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE"] = "3-year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE"] = "4-year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE"] = "5-year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE"] = "6-year"


df["NEW_CONTRACT_LENGTH"] = df["Contract"].apply(lambda x: "yearly" if x in ["One year", "Two year"] else "monthly")


df["NEW_PAYMENT_METHOD"] = df["PaymentMethod"].apply(lambda x: "auto" if x in ['Electronic check',
                                                                                'Bank transfer (automatic)',
                                                                                "Credit card (automatic)"] else "no_auto")

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)


# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)


df.head()
df.shape



# Adım 3: Encoding işlemlerini gerçekleştiriniz.
def grab_col_names(dataframe, cat_th=10, car_th=20):


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# One Hot Encoding

OHE =[col for col in df.columns if 10 >= df[col].nunique() > 2]

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols
df.dtypes

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, OHE, drop_first=True)

df.head()



# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

# StandardScaler:
###################

for col in num_cols:
    ss = StandardScaler()
    df[col + "_standard_scaler"] = ss.fit_transform(df[[col]])
df.head()

# RobustScaler: Medyanı çıkar iqr'a böl.
###################

for col in num_cols:
    rs = RobustScaler()
    df[col + "_robust_scaler"] = rs.fit_transform(df[[col]])

df.head()

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

for col in num_cols:
    mms = MinMaxScaler()
    df[col + "_min_max_scaler"] = mms.fit_transform(df[[col]])
    df.describe().T

df.head()


#############################    Görev 3 : Modelleme      #################################


# Kullanılacak modeller:
# 'LR'
# 'KNN',
# 'CART',
# 'RF'
# 'XGB',
# "GBM"
# "LightGBM",
# "CatBoost"


# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

# df = pd.read_csv("3_Feature_Engineering/datasets/TelcoCustomerChurn-230423-212029.csv")

y = df["Churn"]

X = df.drop(["Churn", "customerID"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:,1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)
# 0.9999825
#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#####################
# CV ile Başarı Değerlendirme
#####################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.73022775662946
cv_results['test_f1'].mean()
# 0.4962242858644152
cv_results['test_roc_auc'].mean()
# 0.657668694729008

###############################################

models = [('LR', LogisticRegression(random_state=17)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=17)),
          ('RF', RandomForestClassifier(random_state=17)),
          ('SVM', SVC(gamma='auto', random_state=17)),
          ('XGB', XGBClassifier(random_state=17)),
          ("LightGBM", LGBMClassifier(random_state=17)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=17))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# ########## LR ##########
# Accuracy: 0.8068
# Auc: 0.8456
# Recall: 0.5452
# Precision: 0.6665
# F1: 0.5995
# ########## KNN ##########
# Accuracy: 0.762
# Auc: 0.7465
# Recall: 0.4414
# Precision: 0.5674
# F1: 0.496
# ########## CART ##########
# Accuracy: 0.7263
# Auc: 0.6554
# Recall: 0.5008
# Precision: 0.4853
# F1: 0.4928
# ########## RF ##########
# Accuracy: 0.7870
# Auc: 0.8249
# Recall: 0.4933
# Precision: 0.6258
# F1: 0.5514
# ########## SVM ##########
# Accuracy: 0.77
# Auc: 0.7205
# Recall: 0.3494
# Precision: 0.6192
# F1: 0.4459
# ########## XGB ##########
# Accuracy: 0.7887
# Auc: 0.8252
# Recall: 0.5147
# Precision: 0.625
# F1: 0.5641
# ########## LightGBM ##########
# Accuracy: 0.8008
# Auc: 0.8351
# Recall: 0.5356
# Precision: 0.653
# F1: 0.5881
# ########## CatBoost ##########
# Accuracy: 0.7999
# Auc: 0.8422
# Recall: 0.5195
# Precision: 0.6558
# F1: 0.5794

####  SEÇİLEN 4 MODEL:

# RF        (F1: 0.5514)
# XGB       (F1: 0.5641)
# LightGBM  (F1: 0.5881)
# CatBoost  (F1: 0.5794)

###  auc skorları en yüksek olan modeller de yine bunlar



# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve
# bulduğunuz hiparparametreler ile modeli tekrar kurunuz.

################################################
# Random Forests
################################################

#
rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Out[197]: 0.7870218810444874--------acc
# Out[198]: 0.5514315372018543--------f1
# Out[199]: 0.8249119449254149--------auc
#


rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_
# {'max_depth': 8, 'max_features': 5,'min_samples_split': 8, 'n_estimators': 200}

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)


cv_results = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#### ilk sonuçlar   /// ikinci sonuçlar ####

# Accuracy: 0.7870  /// 0.8006
# Auc: 0.8249       /// 0.8447
# Recall: 0.4933    /// 0.4911
# Precision: 0.6258 /// 0.6696
# F1: 0.5514        /// 0.5666



################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_best_grid.best_params_
# {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#### ilk sonuçlar   /// ikinci sonuçlar ####

# Accuracy: 0.7887  /// 0.8019
# Auc: 0.8252       /// 0.8418
# Recall: 0.5147    /// 0.5318
# Precision: 0.6250 /// 0.6575
# F1: 0.5641        /// 0.5878



################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
# {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 300}

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#### ilk sonuçlar /// ikinci sonuçlar ####

# Accuracy: 0.8008  /// 0.8043
# Auc: 0.8351       /// 0.8454
# Recall: 0.5356    /// 0.5051
# Precision: 0.6530 /// 0.6766
# F1: 0.5881        /// 0.5779



################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
# {'depth': 3, 'iterations': 500, 'learning_rate': 0.01}

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1","roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#### ilk sonuçlar /// ikinci sonuçlar ####

# Accuracy: 0.7999  /// 0.8059
# Auc: 0.8422       /// 0.8477
# Recall: 0.5195    /// 0.4928
# Precision: 0.6558 /// 0.6887
# F1: 0.5794        /// 0.5740






################################################
# Feature Importance
################################################

def plot_importance(model, features, model_name, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title(f'Feature Importance - {model_name}')  # model_name başlığa eklenir
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig(f'importances_{model_name}.png')  # model_name kaydedilen dosya ismine eklenir


plot_importance(rf_final, X, 'Random Forest')
plot_importance(xgboost_final, X, 'XGBoost')
plot_importance(lgbm_final, X, 'LightGBM')
plot_importance(catboost_final, X, 'CatBoost')


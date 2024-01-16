import numpy
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

df = pd.read_csv("AWSBC7-GR-2/RENTAL-DATA.csv", encoding='unicode_escape')

df.head()
df.info()

df.isnull().sum()
# city                    0
# district                0
# cold_price              0
# object_age            347
# flat_area               1
# room_count             10
# distance_to_centre      5


def grab_col_names(dataframe, cat_th=5, car_th=10):


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


#  Aykırı gözlem var mı ?

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
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


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

    # cold_price True
    # object_age True
    # flat_area True
    # room_count True
    # distance_to_centre True

replace_with_thresholds(df, "object_age")
replace_with_thresholds(df, "room_count")
replace_with_thresholds(df, "distance_to_centre")

df.describe()
df[df["object_age"]== -1] = 0


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


mis_col = missing_values_table(df, True)


# boş değerleri dolduralım:
for col in mis_col:
    df[col] = df[col].fillna(df.groupby('district')['distance_to_centre'].transform('median'))


df["distance_to_centre"] = df["distance_to_centre"].fillna(df["distance_to_centre"].median())

df.isnull().sum()
# BOŞ DEĞER KALMADI

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_but_car, drop_first=True)
df.info()
df.head()

df["city"] = df["city"].astype("string")
df["district"] = df["district"].astype("string")

label_encoder = LabelEncoder()
label_cols = ["city", "district"]
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])

df.head()

# df.to_csv("bu-yeni-dosya.csv")

# Standartlaştırma
# num_cols = [col for col in num_cols if col not in "cold_price" ]
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["cold_price"]
X = df.drop(["cold_price"], axis=1)

###############################################
#####################################################################################################################
from xgboost import XGBRegressor
models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor(random_state = 17)),
          ('RF', RandomForestRegressor(random_state = 17)),
          ('GBM', GradientBoostingRegressor(random_state = 17)),
          ("LightGBM", LGBMRegressor(random_state = 17))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv = 5, scoring = "neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


##########################################################
CART_model = DecisionTreeRegressor(random_state = 1)
CART_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)
               }

CART_gs_best = GridSearchCV(CART_model,
                            CART_params,
                            cv = 3,
                            n_jobs = -1,
                            verbose = True).fit(X, y)
best = CART_gs_best.best_params_

final_model = CART_model.set_params(**best).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv = 5, scoring = "neg_mean_squared_error")))
print(rmse)


random_user = [52, 2, 2, 66, 6, 1]
numnew = np.array(random_user)
numnew22 = numnew.reshape(1, -1)
final_model.predict(numnew22)


final_model = joblib.load("model__2.pkl")
joblib.dump(final_model, "model__2.pkl")





random_user = [52, 211, 2, 66, 2, 6]
numnew = np.array(random_user)
numnew22 = numnew.reshape(1, -1)
new_model = joblib.load("model__2.pkl")

new_model.predict(numnew22)

# sonuç her defasında 7,87 geldi

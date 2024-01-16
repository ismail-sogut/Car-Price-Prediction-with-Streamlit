import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import joblib
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_excel("car price flask/new_car_file3.xlsx")

df.head()

df.info()

for col in df.columns:
    print("##########################################")
    print(col + "_unique_no")
    print(df[col].nunique())
    print("##########################################")


for col in df.columns:
    print("##########################################")
    print(col + "_NO_OF_unique")
    print(df[col].unique())
    print("##########################################")

########### max_power, fuel_consumption ve engine "int" olmalı ya da "float"
########### bu nedenle bhp, kmpl ve CC ifadeleri silinmeli


df["engine"] = df["engine"].str.replace("CC", "", regex=True)
df["engine"] = pd.to_numeric(df["engine"], errors='coerce')

df["fuel_consumption"] = df["fuel_consumption"].str.replace("kmpl", "", regex=True)
df["fuel_consumption"] = pd.to_numeric(df["fuel_consumption"], errors='coerce')

df["max_power"] = df["max_power"].str.replace("bhp", "", regex=True)
df["max_power"] = pd.to_numeric(df["max_power"], errors='coerce')

df.dtypes

df.describe()

df.loc[df["selling_price"]== 1082124, "selling_price"] = 1082
df.loc[df["selling_price"]== 5, "selling_price"] = 5000
df.describe()

df.loc[df["selling_price"]== 8, "selling_price"] = 8000
df.describe()

df.loc[df["selling_price"]== 11, "selling_price"] = 11000
df.describe()

df.loc[df["fuel_consumption"]== 0, "fuel_consumption"] = df["fuel_consumption"].median()
df.describe()



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



# feature eng:

df["NEW_car_age"] = 2023 - df["year"]
df = df.drop(["year"], axis=1)
df = df.drop(["name"], axis=1)

# "Trustmark Dealer" sayısı çok az olduğu için rare ancoding yerine "Dealer"ın içine ekledim
df["seller_type"] = df["seller_type"].str.replace("Trustmark Dealer", "Dealer", regex=True)

# "LPG ve CNG" sayısı çok az olduğu için rare ancoding yerine "Petrol"ın içine ekledim
df["fuel"] = df["fuel"].str.replace("LPG", "Petrol", regex=True)
df["fuel"] = df["fuel"].str.replace("CNG", "Petrol", regex=True)

# "Third Owner, Fourth & Above Owner ve Test Drive Car" sayısı çok az olduğu için rare ancoding yerine "Second Owner"ın içine ekledim
df["owner"] = df["owner"].str.replace("Third Owner", "Second Owner", regex=True)
df["owner"] = df["owner"].str.replace("Fourth & Above Owner", "Second Owner", regex=True)
df["owner"] = df["owner"].str.replace("Test Drive Car", "Second Owner", regex=True)

for col in cat_cols:
    cat_summary(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_col = missing_values_table(df, True)


for col in missing_col:
    df[col] = df[col].fillna(df[col].median())

df.isnull().sum()
# boş değer yok



label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])


######################


y = df["selling_price"]
X = df.drop(["selling_price"], axis=1)


# Splitting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Create and train the model (Random Forest Regressor in this case)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

score = metrics.r2_score(y_test, y_pred)

print("R-squared-error: ", score)


joblib.dump(model, "price_pred.joblib")


# df.columns:
#            ['km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'fuel_consumption', 'engine', 'max_power', 'NEW_car_age']

#     To test the model:
# random_user = [120000, 0, 1, 1, 1, 21.14, 1498, 103.52, 9]
# test_data = np.array(random_user).reshape(1, -1)
# model.predict(test_data)

################################################


#
#
#
#
# #####
# #####                     data/model training
# #####
# #####
#
# # Linear regression
#
# lin_reg_model = LinearRegression()
# lin_reg_model.fit(X_train, y_train)
#
# # model evaluation on train data
#
# y_pred = lin_reg_model.predict(X_train)
#
# error = metrics.r2_score(y_train, y_pred)
#
# print("R-squared-error: ", error)
# # R-squared-error:  0.612859
#
#
# # model evaluation on test data
#
# #############################
#
# y_pred_test = lin_reg_model.predict(X_test)
#
# test_error = metrics.r2_score(y_test, y_pred_test)
#
# print("R-squared-error: ", test_error)
# # R-squared-error:  0.65536429
#

#############################################################
##                     lasso regression
#############################################################


# lasso_model = Lasso()
# lasso_model.fit(X_train, y_train)
#
# # model evaluation on train data
#
# y_pred_lasso = lasso_model.predict(X_train)
#
# error_lasso = metrics.r2_score(y_train, y_pred_lasso)
#
# print("R-squared-error: ", error_lasso)
# R-squared-error:  0.612859139


# model evaluation on test data

#############################

# y_pred_test_lasso = lasso_model.predict(X_test)
#
# test_error_lasso = metrics.r2_score(y_test, y_pred_test_lasso)
#
# print("R-squared-error: ", test_error_lasso)
# # R-squared-error:  0.65382901
#




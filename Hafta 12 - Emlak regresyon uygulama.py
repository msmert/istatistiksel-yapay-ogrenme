#####Emlak regresyon uygulamasi - Hafta 12
import pandas as pd
import numpy as np

# Pandas ayarlarını yapılandırıyoruz: Tüm sütunları göster ve genişliği artır.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Veri Setlerini Yukluyoruz
set1 = pd.read_excel(r"C:\Users\soyka\Desktop\istatistiksel_yapay_ogrenme\emlak_data1.xlsx")
set2 = pd.read_excel(r"C:\Users\soyka\Desktop\istatistiksel_yapay_ogrenme\emlak_data2.xlsx")
df = pd.merge(set1, set2, on='Port_no')

df.describe()

df.head(10)
#    OrtGel  Bina_Yasi    OrtOda     Ort_Y  Nufus  Ort_Hane  Enlem  Boylam  Port_no  Fiyat
# 0  8.3252         41  6.984127  1.023810    322  2.555556  37.88 -122.23      100  4.526
# 1  8.3014         21  6.238137  0.971880   2401  2.109842  37.86 -122.22      101  3.585
# 2  7.2574         52  8.288136  1.073446    496  2.802260  37.85 -122.24      102  3.521
# 3  5.6431         52  5.817352  1.073059    558  2.547945  37.85 -122.25      103  3.413
# 4  3.8462         52  6.281853  1.081081    565  2.181467  37.85 -122.25      104  3.422
# 5  4.0368         52  4.761658  1.103627    413  2.139896  37.85 -122.25      105  2.697
# 6  3.6591         52  4.931907  0.951362   1094  2.128405  37.84 -122.25      106  2.992
# 7  3.1200         52  4.797527  1.061824   1157  1.788253  37.84 -122.25      107  2.414

#Asagida veri seti bagimli ve bagimsiz olarak ikiye ayrilmistir


X = df[["OrtGel", "Bina_Yasi" , "OrtOda","Ort_Y","Nufus", "Ort_Hane","Enlem","Boylam"]]
y = df[['Fiyat']]

X.head(10)
y.head(10)

#Bagimsiz degiskenlerin hepsi normalize edilmistir.
# MinMax scaler

sutun_isim = X.columns
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X, columns=sutun_isim)
X.describe()
# OrtGel     Bina_Yasi        OrtOda         Ort_Y         Nufus      Ort_Hane         Enlem        Boylam       Port_no
# count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000
# mean       0.232464      0.541951      0.032488      0.022629      0.039869      0.001914      0.328572      0.476125      0.500000
# std        0.131020      0.246776      0.017539      0.014049      0.031740      0.008358      0.226988      0.199555      0.288696
# min        0.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
# 25%        0.142308      0.333333      0.025482      0.019943      0.021974      0.001398      0.147715      0.253984      0.250000
# 50%        0.209301      0.549020      0.031071      0.021209      0.032596      0.001711      0.182784      0.583665      0.500000
# 75%        0.292641      0.705882      0.036907      0.022713      0.048264      0.002084      0.549416      0.631474      0.750000
# max        1.000000      1.000000      1.000000      1.000000      1.000000      1.000000      1.000000      1.000000      1.000000


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae

# Scikit-learn ile bir lineer regresyon modeli oluşturuyoruz ve tahminler yaparak ortalama mutlak hata hesaplıyoruz.
model = LinearRegression()
model.fit(X,y)
pred = model.predict(X)
# array([[4.13164983],
#        [3.97660644],
#        [3.67657094],
#        ...,
#        [0.17125141],
#        [0.31910524],
#        [0.51580363]])
mea_validasyonsuz = mae(y, pred)
#0.531164381754645

model1 = pd.DataFrame({'Coef_model1': model.coef_.flatten()}, index=sutun_isim)
model1.loc['mea_validasyonsuz'] = mea_validasyonsuz
#      Coef_model1
# OrtGel                6.332140
# Bina_Yasi             0.481225
# OrtOda              -15.139162
# Ort_Y                21.760216
# Nufus                -0.141874
# Ort_Hane             -4.705313
# Enlem                -3.964568
# Boylam               -4.362518
# mea_validasyonsuz     0.531164


#Veri Setini egitim ve test alt kumelerine ayirmak
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Modeli eğitiriz ve test verileri üzerinde tahminler yaparak ortalama mutlak hata hesaplarız.
model = LinearRegression()
model.fit(X_train,y_train)
pred1 = model.predict(X_test)
# array([[1.92123429],
#        [3.34661488],
#        [2.11122872],
#        ...,
#        [2.32574925],
#        [2.53899645],
#        [1.17416358]])
mea_validasyonlu= mae(y_test, pred1)
#0.5359748760147056

model2 = pd.DataFrame({'Coef_model2': model.coef_.flatten()}, index=sutun_isim)
model2.loc['mea_validasyonlu'] = mea_validasyonlu
#                   Coef_model2
# OrtGel               6.303314
# Bina_Yasi            0.495416
# OrtOda             -14.485712
# Ort_Y               20.734380
# Nufus               -0.174159
# Ort_Hane            -4.318770
# Enlem               -3.970772
# Boylam              -4.357114
# mea_validasyonlu     0.535975

#Katsayilari Karsilastiran Dataframe - iki modeli yan yana gosteriyoruz
model_son = pd.concat([model1, model2], axis=1)
#  Coef_model1  Coef_model2
# OrtGel                6.332140     6.303314
# Bina_Yasi             0.481225     0.495416
# OrtOda              -15.139162   -14.485712
# Ort_Y                21.760216    20.734380
# Nufus                -0.141874    -0.174159
# Ort_Hane             -4.705313    -4.318770
# Enlem                -3.964568    -3.970772
# Boylam               -4.362518    -4.357114
# mea_validasyonsuz     0.531164          NaN
# mea_validasyonlu           NaN     0.535975

#Statsmodel tahmini
import statsmodels.api as sm
# Statsmodels ile bağımsız değişkenleri ekleyerek OLS ile bir lineer regresyon modeli oluşturuyoruz ve özet bilgilerini gösteriyoruz.
X = sm.add_constant(X)
reg_model = sm.OLS(y, X).fit()
reg_model.summary()
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  Fiyat   R-squared:                       0.606
# Model:                            OLS   Adj. R-squared:                  0.606
# Method:                 Least Squares   F-statistic:                     3970.
# Date:                Wed, 31 Jan 2024   Prob (F-statistic):               0.00
# Time:                        15:28:44   Log-Likelihood:                -22624.
# No. Observations:               20640   AIC:                         4.527e+04
# Df Residuals:                   20631   BIC:                         4.534e+04
# Df Model:                           8
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          3.7296      0.067     55.574      0.000       3.598       3.861
# OrtGel         6.3321      0.061    104.054      0.000       6.213       6.451
# Bina_Yasi      0.4812      0.023     21.143      0.000       0.437       0.526
# OrtOda       -15.1392      0.830    -18.235      0.000     -16.766     -13.512
# Ort_Y         21.7602      0.949     22.928      0.000      19.900      23.620
# Nufus         -0.1419      0.169     -0.837      0.402      -0.474       0.190
# Ort_Hane      -4.7053      0.606     -7.769      0.000      -5.892      -3.518
# Enlem         -3.9646      0.068    -58.541      0.000      -4.097      -3.832
# Boylam        -4.3625      0.076    -57.682      0.000      -4.511      -4.214
# ==============================================================================
# Omnibus:                     4393.650   Durbin-Watson:                   0.885
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            14087.596
# Skew:                           1.082   Prob(JB):                         0.00
# Kurtosis:                       6.420   Cond. No.                         320.
# ==============================================================================
# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# """
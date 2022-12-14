import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import altair as alt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.write("""
# Power factor prediction App for three phase systems
This app can predict the power factor variations in three phase installations thus giving important information about the presence of non linear or
strongly inductive loads.
""")
st.sidebar.header('Please input data for analysis')

# st.sidebar.markdown("""
# [Cargar el archivo de ejemplo](https://raw.githubusercontent.com/jorgetorre70/PowerFactor-prediction/main/ALSA_correlation.csv)
# """)

# Collects user input features into dataframe


uploaded_file = st.sidebar.file_uploader("Your file should be in CSV format", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def load_data():
        dfcorr3 = pd.read_csv('https://raw.githubusercontent.com/jorgetorre70/PowerFactor-prediction/main/Yerbabuena_correlationbis.csv',sep=',')
        return dfcorr3

    input_df = load_data()

# Displays the user input features
st.subheader('User features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Please notice that your CSV file should have only 3 columns (one for each of the 3 phase currents) according the following example:')
    st.write(input_df)

#Scaling data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(input_df.iloc[:, 0:3]))
df_unscaled = input_df.iloc[:, 0:3]

# Reads in saved classification model
load_RFmodel = pickle.load(open('RFmodel.pkl', 'rb'))

# Apply model to make predictions
st.subheader('Prediction')

st.subheader('Model metrics')
y_real = input_df.iloc[:, [3]].values
y_predict = load_RFmodel.predict(input_df.iloc[:, 0:3])

MAE_result = mean_absolute_error(y_real,y_predict)
MSE_result = mean_squared_error(y_real,y_predict)
RMSE_result = round(np.sqrt(MSE_result),3)
R2score_result = round(r2_score(y_pred=y_predict,y_true=y_real),2)
datos = {'Random Forest Model':[MAE_result,MSE_result, RMSE_result]}
df = pd.DataFrame(data=datos, index=pd.Index(['MAE','MSE','RMSE']))

st.write(df)

# dfgraf = pd. DataFrame()
# dfgraf['ejex']= np.linspace(0,len(y_real))
# dfgraf['real'] = input_df['PF'].copy()
# dfgraf['pred'] = pd.Series(prediction)

st.subheader('Plot')

my_labels = {"x1" : "Actual values", "x2" : "Predicted values"}

xgraf1 = np.linspace(0,9985,1997)
fig = plt.figure(figsize=(10,5), dpi=100)
fig.suptitle('Prediction using RF model', fontsize=16)
graficar = pd.DataFrame(data=xgraf1, columns=['index'])
graficar['PF'] = input_df['PF']
graficar['predict'] = pd.Series(y_predict)
sns.lineplot(x=xgraf1,y=graficar.PF,color='b',linewidth = 0.9,label = my_labels["x1"]).set( xlabel = "time(mins)", ylabel = "Power Factor")
sns.lineplot(x=xgraf1,y=(graficar.predict),color='r',linewidth = 0.9,label = my_labels["x2"])
plt.legend(loc='upper right')
plt.xlim(0)
plt.tight_layout()
st.pyplot(fig)

st.caption('**?? Jorge de la Torre y Ramos - 2022**')
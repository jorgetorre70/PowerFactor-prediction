import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import altair as alt
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns


st.write("""
# Power Factor Prediction App
This app predicts the power factor variations in a three phase system
""")
st.sidebar.header('User Input Data')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/jorgetorre70/PowerFactor-prediction/main/ALSA_correlation.csv)
""")

# Collects user input features into dataframe


uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def load_data():
        dfcorr3 = pd.read_csv('https://raw.githubusercontent.com/jorgetorre70/PowerFactor-prediction/main/Yerbabuena_correlationbis.csv',sep=',')
        return dfcorr3

    input_df = load_data()

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

#Scaling data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(input_df.iloc[:, 0:3]))
df_unscaled = input_df.iloc[:, 0:3]

# Reads in saved classification model
load_RFmodel = pickle.load(open('RFmodel.pkl', 'rb'))

# Apply model to make predictions
dfpred = df_unscaled
prediction = load_RFmodel.predict(dfpred)

y_real = input_df.iloc[:, [3]].values
MSE_1st = mean_squared_error(y_real,prediction)
RMSE_1st = np.sqrt(MSE_1st)
r2score_1st = round(r2_score(y_pred=prediction,y_true=y_real),2)
datos = {'Random Forest':[RMSE_1st,r2score_1st]}
df = pd.DataFrame(data=datos, index=pd.Index(['RMSE', 'R2']))
st.subheader('Métricas del modelo')
st.write(df)

# dfgraf = pd. DataFrame()
# dfgraf['ejex']= np.linspace(0,len(y_real))
# dfgraf['real'] = input_df['PF'].copy()
# dfgraf['pred'] = pd.Series(prediction)
st.subheader('Prediction')
y_predict = load_RFmodel.predict(input_df.iloc[:, 0:3])

st.subheader('Gráfica')

xgraf1 = np.linspace(0,9985,1997)
fig = plt.figure(figsize=(10,5), dpi=100)
fig.suptitle('Predicción con modelo RF', fontsize=16)
graficar = pd.DataFrame(data=xgraf1, columns=['index'])
graficar['PF'] = input_df['PF']
graficar['predict'] = pd.Series(y_predict)
sns.lineplot(x=graficar.index,y=graficar.PF, label='ELC-3').set( xlabel = "time(mins)", ylabel = "Power Factor")
sns.lineplot(x=graficar.index,y=graficar.predict, label='predict').set( xlabel = "time(mins)", ylabel = "Power Factor")
plt.legend(['Actual values', 'Predicted values'],loc='upper right')
st.pyplot(fig)



# a = alt.Chart(dfgraf).mark_line().encode(
#     x=alt.X('ejex', axis=alt.Axis(title='Tiempo(mins)')),
#     y=alt.Y('real', axis=alt.Axis(title='Factor de potencia')))

# b = alt.Chart(dfgraf).mark_line(color="#F52407").encode(
#     x='ejex', y='pred')

# c = alt.layer(a,b)

# st.altair_chart(c, use_container_width=True)

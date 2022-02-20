from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import plotly.figure_factory as ff
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode,iplot
import streamlit as st 
import chart_studio.plotly as py
import plotly.graph_objects as go

import plotly.express as px

st.set_page_config(layout="wide")

st.title("Welcome...")
st.markdown("""**Hello**  and **welcome** to my first streamlit application in MSBA325 course!

In this application, I used two datasets. The first is: **graduate admission prediction** to study the factors that increase the chance for graduate studies admission. 
The second is **life expectancy** to see the difference in age expectancy among countries.""")

Graduate_admission= ' [Graduate Admission](https://www.kaggle.com/mohansacharya/graduate-admissions)'
Life_Expectancy=  '  [Life Expectancy](https://www.kaggle.com/amansaxena/lifeexpectancy)'
st.markdown("""These datasets are obtained from kaggle and can be checked using the following links:""")
st.markdown(Graduate_admission,unsafe_allow_html=True)
st.markdown(Life_Expectancy,unsafe_allow_html=True)

st.title("Graduate Admission Prediction")
from PIL import Image
image = Image.open(r"C:\Users\BC\Downloads\photo1.jpg")
st.image(image)
df = pd.read_csv(r"C:\Users\BC\Downloads\Admission_Predict.csv")

if st.checkbox('Show graduate admission data'):
      st.subheader('Graduate Admission Data')
      st.write(df)

st.title("Choose a Score")
pages_names = ("CGPA","TOEFL Score", "GRE Score")
page=st.radio('Navigation',pages_names)


st.write("Chance of Admit by ", page)
if page == "CGPA" :
 ax=px.scatter(df,x="CGPA",y="Chance of Admit ", color="LOR ", size_max=10, hover_name="Serial No.")
 st.plotly_chart(ax)
if page =="TOEFL Score" :
    ax=px.scatter(df,x="TOEFL Score",y="Chance of Admit ", color="LOR ", size_max=10, hover_name="Serial No.")
    st.plotly_chart(ax)
if page ==  "GRE Score" :
    ax=px.scatter(df,x="GRE Score",y="Chance of Admit ", color="LOR ", size_max=10, hover_name="Serial No.")
    st.plotly_chart(ax)

st.subheader(f'Scatter Plot showing the chance of graduate admission by CGPA and University Rating ')
df["sum of grades"]=df["GRE Score"]+df["TOEFL Score"]+df["CGPA"]
ax1=px.scatter(df,x="CGPA",y="Chance of Admit ", color="LOR ",size="TOEFL Score", size_max=10, hover_name="Serial No.",facet_col="University Rating")
st.plotly_chart(ax1)


st.subheader(f'Animated Scatter Plot showing the correlation between TOEFL Score and GRE  Score ')
ax2=px.scatter(df,x="GRE Score",y="TOEFL Score", color="CGPA", hover_name="Serial No.",animation_frame="LOR ")
st.plotly_chart(ax2)

University_Rating = st.slider('University Rating', 1, 5, 3)
st.subheader(f'Chance of Graduate Admission by University Rating = {University_Rating}')
axx=px.bar(x=df["University Rating"]== University_Rating, y= df["Chance of Admit "])
st.plotly_chart(axx)

st.subheader(f'Histogram showing the chance of admit for graduate by research and sum of grades')
ax3=px.histogram(df,x="sum of grades",y="Chance of Admit ", histfunc="avg",color="Research")
st.plotly_chart(ax3)

st.subheader(f'Box Plot showing the chance of admit for graduate by research')
ax4=px.box(df,x="Research",y="Chance of Admit ",color="Research")
st.plotly_chart(ax4)

st.title("Life Expectancy")
image = Image.open(r"C:\Users\BC\Downloads\photo2.jpg")
st.image(image)
df1=pd.read_csv(r"C:\Users\BC\Downloads\Life_expectancy_dataset.csv",encoding='latin-1')

if st.checkbox('Show Life Expectancy Data'):
    st.subheader('Life Expectancy Data')
    st.write(df1)

import plotly.express as px
st.subheader("3d Scatter Plot Showing the Overall Life by Country and Continent")
fig = px.scatter_3d(df1, x='Country', y='Continent', z='Overall Life',
              color='Continent', size='Overall Life', size_max=15,
               opacity=0.7)
ax=fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(ax)

st.subheader("Sunburst showing the overall life in each country in the continents")
fig=px.sunburst(df1, color="Overall Life", values="Overall Life", path=["Continent","Country"],hover_name="Country")
ax2=fig.update_layout(
    margin = dict(t=25, l=30, r=25, b=30)
)
st.plotly_chart(ax2)

df2=df1.groupby("Continent",as_index=False)[["Overall Life","Male Life", "Female Life"]].mean()


st.subheader("Life Expectancy by Continent")
st.sidebar.checkbox("Show Analysis by Continent", True, key=1)
select = st.sidebar.selectbox('Select a Continent',df2['Continent'])

#get the state selected in the selectbox
state_data = df2[df2['Continent'] == select]
select_status = st.sidebar.radio("Life Expectancty by Continent", ('Overall Life',
'Male Life', 'Female Life',))

def get_total_dataframe(dataset):
    total_dataframe = pd.DataFrame({
    'Status':['Overall Life', 'Male Life', 'Female Life'],
    'Number of cases':(dataset.iloc[0]['Overall Life'],
    dataset.iloc[0]['Male Life'], 
    dataset.iloc[0]['Female Life'])})
    return total_dataframe
state_total = get_total_dataframe(df2)

if st.sidebar.checkbox("Show Analysis by Continent", True, key=2):
    st.markdown("## **Continent level analysis**")
    st.markdown("### Overall Life, Male Life and, Female Life " +
    " in %s " % (select))
    if not st.checkbox('Hide Graph', False, key=1):
        state_total_graph = px.bar(
        state_total, 
        x='Status',
        y='Number of cases',
        labels={'Number of cases':'Number of cases in %s' % (select)},
        color='Status')
        st.plotly_chart(state_total_graph)



st.sidebar.write("**Contact** **Details:**")
st.sidebar.write("Done by: **Ghida** **Raydan**")
st.sidebar.write("email:gmr07@mail.aub.edu")





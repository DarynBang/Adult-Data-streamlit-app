import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.set_page_config(layout='wide')
st.markdown(hide_footer_style, unsafe_allow_html=True)

#Import data here
data = pd.read_csv(r"https://raw.githubusercontent.com/DarynBang/Adult-Data-streamlit-app/master/adult.csv")
sns.set_style("dark")
df = data.copy()

def plotly_layout(figure):
    figure.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(figure)


st.title("Adult data Project")
 
st.sidebar.title("Sidebar Display")
options = st.sidebar.radio("Select Info below", options=['Statistics', 'Visualizations'])
#Removing irrelevant features
df.drop(['fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'marital-status'], axis=1, inplace=True)

df.replace("?", np.nan, inplace=True)

info = pd.DataFrame(df.isnull().sum(), columns=["Is Null"])
info.insert(1, value=df.duplicated().sum(), column='Duplicated', allow_duplicates=True)
info.insert(2, value=df.nunique(), column='Unique', allow_duplicates=True)
info.insert(3, value=df.dtypes, column='Dtype', allow_duplicates=True)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

categorical_features = df.select_dtypes(include=[object]).columns
numerical_features = df.select_dtypes(exclude=[object]).columns

income_cat = pd.get_dummies(df['income'], drop_first=True, dtype=int)
gender = pd.get_dummies(df['gender'], drop_first=True, dtype=int)

fixed_df = pd.concat([df.drop(["income", 'gender'], axis=1), income_cat, gender], axis=1)

fixed_features = fixed_df.select_dtypes(include=[object]).columns
from category_encoders import TargetEncoder
encoder = TargetEncoder()
for col in fixed_features:
    fixed_df[col] = encoder.fit_transform(fixed_df[col].values.reshape(-1, 1), fixed_df['>50K'])

X = fixed_df.drop(">50K", axis=1)
y = fixed_df['>50K']

st.session_state[['X', 'y']] = X, y

if options == 'Statistics':
    st.subheader("Data Statistics")
    st.markdown("----")
    st.markdown("Data Monitoring")
    st.write(data.head(10))

    st.markdown('----')
    st.markdown("Data Information")
    st.write(info)

    st.markdown("----")
    st.markdown("Numeric data")
    st.write(df.describe())

elif options == 'Visualizations':
    st.subheader("Data Visualiazations")
    st.markdown("----")
    st.markdown('Cleaned data')
    st.write(df.head(10))

    st.markdown('-----')
    st.markdown("Numeric data visualization")
    #Numeric data histplots
    n_bins = st.slider("Select number of bins", min_value=10, max_value=50, value=30, step=2)
    dist_c1, dist_c2 = st.columns((5, 5))
    with dist_c1:
        ax = sns.histplot(kde=True, bins=n_bins, data=df, x='age', line_kws={'lw': 3})
        ax.lines[0].set_color('crimson')
        age_hist = plt.gcf()
        ax.set_title('Age distribution', fontdict={'weight': 'bold', 'color': 'grey'}, loc='left')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        plt.tight_layout()
        plt.xlabel(None)
        plt.ylabel(None)
        st.pyplot(age_hist)

    with dist_c2:
        plt.clf()
        ax = sns.histplot(kde=True, bins=n_bins, x='hours-per-week', data=df, line_kws={'lw': 3})
        ax.lines[0].set_color('crimson')
        hpw_hist = plt.gcf()
        ax.set_title('Hours of work Distribution', fontdict={'weight': 'bold', 'color': 'grey'}, loc='left')
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        plt.tight_layout()
        plt.xlabel(None)
        plt.ylabel(None)
        st.pyplot(hpw_hist)


    st.markdown("-----")
    st.markdown("Categorical Data Monitoring")

    barh_c1, barh_c2 = st.columns((4, 4))
    with barh_c1:
        labels1 = st.selectbox("Select category to monitor", options=categorical_features)
        fig1 = px.bar(data_frame=df[labels1].value_counts(ascending=True).tail(8), orientation='h',
                      title=f"{labels1.capitalize()} data monitor", width=600)
        plotly_layout(fig1)

    with barh_c2:
        labels2 = st.selectbox("Select another category to monitor", options=categorical_features)
        fig2 = px.bar(data_frame=df[labels2].value_counts(ascending=True).tail(8), orientation='h', width=600,
                      title=f"{labels1.capitalize()} data monitor")
        plotly_layout(fig2)

    category_data = df[categorical_features].nunique().sort_values()
    cat_bar = px.bar(data_frame=category_data, text_auto=True, orientation='h', title='Categories')
    cat_bar.update_traces(textposition='outside', marker_color='gray')
    plotly_layout(cat_bar)


    st.markdown("-----")
    st.markdown("Correlation Matrix")

    corr = fixed_df.corr()
    plt.clf()
    sns.heatmap(data=corr, linewidth=0.2, annot=True, fmt='.2g',
                mask=np.triu(np.ones_like(corr, dtype=bool)), vmax=True,
                cmap='magma')
    corr_heatmap = plt.gcf()
    st.pyplot(corr_heatmap)

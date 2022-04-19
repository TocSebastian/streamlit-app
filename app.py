import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Movies by Rating")

df = pd.read_pickle('df8k.pkl')
df = df.set_index('name').reset_index()
df['metascore'] = pd.to_numeric(df['metascore'])
list_set_actors = pd.read_pickle('list_set_actors.pkl')
list_set_directors = pd.read_pickle('list_set_directors.pkl')

# movie_name = st.sidebar.selectbox('Alege un film',df.name.unique().tolist())
# (df.name == movie_name)
metascore_colors = ['red', 'yellow', 'green']
metascore_dict = {'red':range(0,40),'yellow':range(40,61),'green':range(61,101),'all':range(0,101)}
sort_valori = ['nota','metascore','year','votes','duration_numerical']


# rated = st.sidebar.multiselect("Alege R",df.R.unique().tolist())

title_sidebar = st.sidebar.header("Filtre")


x = st.sidebar.slider('Alege un an', int(min(df.year.unique())), int(max(df.year.unique())),(1990,2010))

# --------------------------------------------------------------------------------
metascore_container = st.sidebar.container()                                    #
all_meta = st.sidebar.checkbox("Select all metascore")                          #
if all_meta:                                                                    #
    metascore = 'all'                                                           #  ---> metascore sidebar widget
else:                                                                           #
    metascore = st.sidebar.select_slider('Metascore',metascore_colors)          #
                                                                                #
color = metascore_dict[metascore]                                               #
# --------------------------------------------------------------------------------
sortare = st.sidebar.multiselect('Sorteaza dupa:',sort_valori,'nota')


actors_sidebar =  st.sidebar.checkbox('Actor')
if actors_sidebar:
    actors = st.sidebar.selectbox("Alege actor:",list_set_actors)


director_sidebar = st.sidebar.checkbox('Director')
if director_sidebar:
    director = st.sidebar.selectbox('Alege un director:',list_set_directors)


rated_container = st.sidebar.container()
all = st.sidebar.checkbox("Select all")
if all:
    rated = rated_container.multiselect("Alege R",df.R.unique().tolist(),df.R.unique().tolist())
else:
    rated = rated_container.multiselect("Alege R",df.R.unique().tolist(),'R')



df_years = df.loc[(df.year>=x[0]) & (df.year<=x[1])&(df.R.isin(rated)) & (df.metascore >= (min(color))) & (df.metascore <=max(color))]
df_final= df_years.sort_values(sortare, ascending = False)

if actors_sidebar:
    df_final = df_final[df_final.actors.apply(lambda x: x.count(actors) > 0 if x is not None else False)]

if director_sidebar:
    df_final = df_final[df_final.director.apply(lambda x: x.count(director) > 0 if x is not None else False)]

st.write(df_final)

df_csv = df_final.to_csv()

st.download_button('Download CSV here',df_csv)






col1,col2,col3,col4 = st.columns(4)

with col1:
    x_label = st.selectbox('Ox',['year'])
with col2:
    y_label = st.selectbox('Oy',['nota','metascore'])
with col3:
    function = st.selectbox('Functions',['mean','max','min'])
with col4:
    numb_years = st.selectbox('Pass',['1 year','5 years','10 years','25 years'])
    numb_years_dict = {'1 year':1,'5 years':5,'10 years':10,'25 years':25}
    df_final['year'] = df_final['year'] - df_final['year'] % numb_years_dict[numb_years]




df_plot = df_final.groupby(x_label,as_index=False).agg({y_label:function})

def barPlot():
    fig = plt.figure(figsize=(12,4))
    sns.barplot(x =x_label, y = y_label, data = df_plot)
    plt.xticks(rotation = 90)
    st.pyplot(fig)

def linePlot():
    fig = plt.figure(figsize=(10,4))
    sns.lineplot(x =x_label, y = y_label, data = df_plot)
    plt.xticks(rotation = 45)
    st.pyplot(fig)


def execute_graph(graph_type):
    return {'LinePlot': lambda : linePlot(),
                  'BarPlot': lambda : barPlot()
    }[graph_type]()

graph_type = st.selectbox("Alege un tip de grafic",['LinePlot','BarPlot'])
graph_type
execute_graph(graph_type)

dist_type = st.selectbox('DistPlot',['year','nota','metascore'])
def distPlot():
    fig = plt.figure(figsize=(10,4))
    sns.distplot(df_final[dist_type])
    plt.xticks(rotation = 45)
    st.pyplot(fig)
distPlot()
# st.color_picker('Pick a color')



















# container = st.sidebar.container()
# aicea = st.sidebar.checkbox("Select all1")
#
# if aicea:
#     selected_options = container.multiselect("Select one or more options:",
#          ['A', 'B', 'C'],['A', 'B', 'C'])
# else:
#     selected_options =  container.multiselect("Select one or more options:",
#         ['A', 'B', 'C'])

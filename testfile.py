import streamlit as st
import hydralit_components as hc
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_css import local_css
from PIL import Image
from scipy import stats
import numpy as np
import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle5 as pickle
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


img = Image.open('logo.svg.webp')

#make it look nice from the start
st.set_page_config(page_title = 'Movies IMDb',page_icon=img, layout='wide',initial_sidebar_state='collapsed',)

# specify the primary menu definition
menu_data = [
    {'icon': "bi bi-camera-reels-fill", 'label':"Movies Dataframe"},
    {'id':'Movie','icon':"bi bi-film",'label':"Movie"},
    # {'icon': "fa-solid fa-radar",'label':"Dropdown1", 'submenu':[{'id':' subid11','icon': "fa fa-paperclip", 'label':"Sub-item 1"},{'id':'subid12','icon': "💀", 'label':"Sub-item 2"},{'id':'subid13','icon': "fa fa-database", 'label':"Sub-item 3"}]},
    # {'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
    # {'id':' Crazy return value 💀','icon': "💀", 'label':"Calendar"},
     # {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
    # {'icon': "far fa-copy", 'label':"Right End"},
     # {'icon': "fa-solid fa-radar",'label':"Dropdown2", 'submenu':[{'label':"Sub-item 1", 'icon': "fa fa-meh"},{'label':"Sub-item 2"},{'icon':'🙉','label':"Sub-item 3",}]},
]

over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    # login_name='Logout',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=False, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)
# ------------------------------------------------------------------------------
if menu_id == 'Movies Dataframe':
    st.title("Movies by Rating")
    with open('df8k.pkl', "rb") as fh:
        df = pickle.load(fh)
                                               # incarcare dataframe precum si 2 liste cu actori si directori de filmi fara duplicate
    df = df.set_index('name').reset_index()
    df['metascore'] = pd.to_numeric(df['metascore'])



    with open('list_set_actors.pkl', "rb") as fh:
        list_set_actors = pickle.load(fh)






    with open('list_set_directors.pkl', "rb") as fh:
        list_set_directors= pickle.load(fh)

# ------------------------------------------------------------------------------
    # movie_name = st.sidebar.selectbox('Alege un film',df.name.unique().tolist())
    # (df.name == movie_name)

# -------------------------------------------------------------------------------------------------------
    metascore_colors = ['red', 'yellow', 'green']
    metascore_dict = {'red':range(0,40),'yellow':range(40,61),'green':range(61,101),'all':range(0,101)}   # dictionar pentru metascore
    sort_valori = ['nota','metascore','year','votes','duration_numerical']                                # parametrii dupa care vom sorta dataframe-ul
# -------------------------------------------------------------------------------------------------------

    # rated = st.sidebar.multiselect("Alege R",df.R.unique().tolist())
# --------------------------------------------------------------------------------------------------------------
    title_sidebar = st.sidebar.header("Filtre")                                                                 # titlu Sidebar


    x = st.sidebar.slider('Alege un an', int(min(df.year.unique())), int(max(df.year.unique())),(1990,2010))    # slider pentru ani

# ----------------------------------------------------------------------------------
    metascore_container = st.sidebar.container()                                    #
    all_meta = st.sidebar.checkbox("Select all metascore")                          #
    if all_meta:                                                                    #
        metascore = 'all'                                                           #  ---> metascore sidebar widget
    else:                                                                           #
        metascore = st.sidebar.select_slider('Metascore',metascore_colors)          #
                                                                                    #
    color = metascore_dict[metascore]                                               #
# ----------------------------------------------------------------------------------
    sortare = st.sidebar.multiselect('Sorteaza dupa:',sort_valori,'nota')           # alegeam parametrul dupa care vom sorta
#-----------------------------------------------------------------------------------

    actors_sidebar =  st.sidebar.checkbox('Actor')
    if actors_sidebar:
        actors = st.sidebar.selectbox("Alege actor:",list_set_actors)              # in cazul in care bifam actors sau director putem sorta in functie de acestia


    director_sidebar = st.sidebar.checkbox('Director')
    if director_sidebar:
        director = st.sidebar.selectbox('Alege un director:',list_set_directors)
# ----------------------------------------------------------------------------------

    rated_container = st.sidebar.container()
    all = st.sidebar.checkbox("Select all")
    if all:
        rated = rated_container.multiselect("Alege R",df.R.unique().tolist(),df.R.unique().tolist())    # contaire unde putem selecta valoare pentru rated
    else:                                                                                               # si de unde putem selecta toate optiunile
        rated = rated_container.multiselect("Alege R",df.R.unique().tolist(),'R')

# ------------------------------------------------------------------------------------------------------

    df_years = df.loc[(df.year>=x[0]) & (df.year<=x[1])&(df.R.isin(rated)) & (df.metascore >= (min(color))) & (df.metascore <=max(color))]    # aici are loc sortarea in functie de parametrii de mai sus inafara de actori si directori
    df_final= df_years.sort_values(sortare, ascending = False)
# ---------------------------------------------------------------------------------------------------------------------------------------------
    if actors_sidebar:
        df_final = df_final[df_final.actors.apply(lambda x: x.count(actors) > 0 if x is not None else False)]
                                                                                                                                               # sortarea in functie de actori si directori
    if director_sidebar:
        df_final = df_final[df_final.director.apply(lambda x: x.count(director) > 0 if x is not None else False)]
# ---------------------------------------------------------------------------------------------------------------------------------------------
    def color_df(val):
        if val > 60:
            color = 'green'
        elif val > 39:
            color = 'orange'                                                    # functie pentru a colora valorile din coloana metascore in functie de valoarea lor
        elif val > 0:
            color = 'red'
        return f'background-color: {color}'
    st.dataframe(df_final.style.applymap(color_df,subset = ['metascore']))
    # st.write(df_final)
# -------------------------------------------------------------------------------
    df_csv = df_final.to_csv()
                                                                                # convertim dataframe-ul in csv si oferim posibilitatea de al downloada
    st.download_button('Download CSV here',df_csv)

# -------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------
    col1,col2,col3,col4 = st.columns(4)

    with col1:
        x_label = st.selectbox('Ox',['year'])
    with col2:
        y_label = st.selectbox('Oy',['nota','metascore'])
    with col3:                                                                                   # cele 4 optiuni pentru afisarea graficelor
        function = st.selectbox('Functions',['mean','max','min'])
    with col4:
        numb_years = st.selectbox('Pass',['1 year','5 years','10 years','25 years'])
        numb_years_dict = {'1 year':1,'5 years':5,'10 years':10,'25 years':25}
        df_final['year'] = df_final['year'] - df_final['year'] % numb_years_dict[numb_years]

# -----------------------------------------------------------------------------------------------


    df_plot = df_final.groupby(x_label,as_index=False).agg({y_label:function})                   # dataframe-ul care urmeaza a fi afisat in funtie de valorile optiunilor de mai sus(cele 4)

# -----------------------------------------------------------------------------------------------
    def barPlot():
        fig = plt.figure(figsize=(12,4))
        sns.barplot(x =x_label, y = y_label, data = df_plot)                                     # functie pentru a afisa BarPlot
        plt.xticks(rotation = 90)
        st.pyplot(fig)
# -----------------------------------------------------------------------------------------------
    def linePlot():
        fig = plt.figure(figsize=(10,4))
        sns.lineplot(x =x_label, y = y_label, data = df_plot)                                    # functie pentru a afisa LinePlot
        plt.xticks(rotation = 45)
        st.pyplot(fig)
# -----------------------------------------------------------------------------------------------

    def execute_graph(graph_type):
        return {'LinePlot': lambda : linePlot(),                                                 # functie care face plotul in functie de tipul de plot ales
                      'BarPlot': lambda : barPlot()
        }[graph_type]()
# -----------------------------------------------------------------------------------------------
    graph_type = st.selectbox("Alege un tip de grafic",['LinePlot','BarPlot'])
    graph_type                                                                                   # alegem un tip de plot si il afisam
    execute_graph(graph_type)
# -----------------------------------------------------------------------------------------------
    dist_type = st.selectbox('DistPlot',['year','nota','metascore'])
    def distPlot():
        fig = plt.figure(figsize=(10,4))
        sns.distplot(df_final[dist_type])                                                        # afisam un Distribution Plot in functie de parametrul ales
        plt.xticks(rotation = 45)
        st.pyplot(fig)
    distPlot()
# -----------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
elif menu_id == "Movie":

    with open('df8k.pkl', "rb") as fh:
        df = pickle.load(fh)



    with open('all_movies_images_links.pkl', "rb") as fh:
        movie_links_images = pickle.load(fh)
                                                                                        # pagina noua

    df["images_links"] = movie_links_images                                     # incarcam datafraem-ul, lista cu linkuri ale imaginilor si combinam in acelasi dataframe

    with open('list_movie_names_years.pkl', "rb") as fh:
        movies = pickle.load(fh)
                                                                                # incarcam lista cu nume_film + an pentru a putea cauta filmele corespunzator

    with open('Top8kMovies.pkl', "rb") as fh:
        movies_with_links = pickle.load(fh)




    df['movie_link'] = movies_with_links['link']
    del df['index']



    # st.set_page_config(layout="wide")

# -------------------------------------------------------------------------------

    movie_year = st.selectbox("Alege un film",movies)

    movie_splitted = movie_year.rsplit(' ',1)                                   # impartim filmele din select box in nume si in an
    movie = movie_splitted[0]
    year_movie = int(movie_splitted[1])

# -------------------------------------------------------------------------------
    st.title(movie)

    movie_df = df[(df.name == movie) & (df.year == year_movie)]
# ------------------------------------------------------------------------------
    yrd = movie_df[['year','R','duration']]

    year = yrd['year'].values[0]
    rated = yrd['R'].values[0]                                                  # selectam valorile pentru year, rated si duration din movie_df
    duration = yrd['duration'].values[0]

    yrd_output = str(str(year) + ' ' + str(rated) + ' ' + str(duration))
# ------------------------------------------------------------------------------

    # st.session_state.emoji = "⭐"
# ------------------------------------------------------------------------------- html pentru a afisa bolduit year, rated si duration
    html_yrd = f"""
    <style>
    p.a {{
      font: bold 20px Courier;
    }}
    </style>
    <p class="a">{year}  {rated}  {duration} </p>
    """
# -------------------------------------------------------------------------------html pentru a afisa bolduit nota filmului
    html_rating = f"""
    <style>
    p.a {{
      font: bold 20px Courier;
    }}
    </style>
    <p class="a">⭐ {movie_df['nota'].values[0]}/10 </p>
    """
# -------------------------------------------------------------------------------html pentru a afisa bolduit numarul de voturi al filmului

    html_votes = f"""
    <style>
    p.a {{
      font: bold 20px Courier;
    }}
    </style>
    <p class="a">{movie_df['votes'].values[0]} votes</p>
    """
# -------------------------------------------------------------------------------
    def f_str(x,sep = ' '):
        l=''
        for i in x:                                                             #separa elementele din lista pentru a le pune impreuna ca string si a le afisa
            l = l + i + sep
        return l

# -------------------------------------------------------------------------------
    local_css("style.css")
    def color_metascore(x):
        if x > 60:
            color = 'green'
        elif x > 40:                                                            # functie folosita pentru a afla culoarea in funtie de metascore
            color = 'yellow'
        else:
            color = 'red'
        return color
# -------------------------------------------------------------------------------


    col1,col2,col3 = st.columns([1,3,1])
# -------------------------------------------------------------------------------
    with col1:
        st.markdown(html_yrd, unsafe_allow_html=True)                           # aici afisam prima coloana adica cea unde avem anul,rated si durata
        st.image(movie_df['images_links'].values[0])                            # precum si imaginea filmului
# -------------------------------------------------------------------------------
    with col2:
        st.header('Details')
        try:
            st.write('Genres:', f_str(movie_df['genres'].values[0]))
        except:
            st.write('Genres:', None )                                          # aici afisam toate informatiile si folosim functia 'f_str' pentru
        st.write('Description')                                                 # a putea afisa sub forma de string valorile din coloane
        st.write(movie_df['description'].values[0])
        st.write('Actors:', f_str(movie_df['actors'].values[0],', ')[:-2])
        st.write('Director:', movie_df['director'].values[0])
        st.write('Writers:', f_str(movie_df['actors'].values[0],', ')[:-2])
        try:
            st.write('Tagline:', movie_df['tagline'].values[0])
        except:
            st.write('Tagline:', None )

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    with col3:

        st.markdown(html_rating, unsafe_allow_html = True)
        st.markdown(html_votes, unsafe_allow_html = True)
        st.markdown('#')
        st.markdown('#')                                                        # aici afisam nota si numarul de voturi folosind functiile html de mai devreme si deasemenea afisam si metascorul
        st.markdown('#')                                                        # intr-un mod highlithed in fucntie de valoare folosind functia de mai sus 'color_metascore' precum si f string
        st.markdown('#')
        st.markdown('#')

        try:
            t = f"""<div><span class='highlight {color_metascore(int(movie_df['metascore'].values[0]))}'> <span class='bold'>{movie_df['metascore'].values[0]}</span> </span><span class='bold'>Metascore</span></div>"""
        except:
            t = f"""<div><span class='highlight grey'><span class='bold'>No</span></span><span class='bold'> Metascore</span></div> """
        st.markdown(t, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    movie_df

    # movie_id = df[(df.name ==  movie) & (df.year == year_movie) ]['index'].values[0]
    # movie_link = movies_with_links[movies_with_links.index == movie_id]['link'].values[0]
    # movie_link
    # movies_with_links
    r = requests.get(movie_df['movie_link'].values[0])
    doc = BeautifulSoup(r.content, "html.parser")

    images_movies = doc.find_all(class_="ipc-media ipc-media--avatar ipc-image-media-ratio--avatar ipc-media--base ipc-media--avatar-m ipc-media--avatar-circle ipc-avatar__avatar-image ipc-media__img")[0:7]

    dict_actors = {'name':[],'img_link':[]}
    for x in images_movies:
        dict_actors['name'].append(x.img['alt'])
        dict_actors['img_link'].append(x.img['src'])
    actors_images = pd.DataFrame(dict_actors)
    actors_images
#     x.img['srcset'].split(',')

    st.subheader('Actors')

    col_movie1,col_movie2,col_movie3,col_movie4,col_movie5,col_movie6,col_movie7 = st.columns(7)                                                                                                                                  # Actori care joaca in film

    with col_movie1:
        st.write(actors_images.iloc[0,0])
        st.image(actors_images.iloc[0,1])
    with col_movie2:
        st.write(actors_images.iloc[1,0])
        st.image(actors_images.iloc[1,1])
    with col_movie3:
        st.write(actors_images.iloc[2,0])
        st.image(actors_images.iloc[2,1])
    with col_movie4:
        st.write(actors_images.iloc[3,0])
        st.image(actors_images.iloc[3,1])
    with col_movie5:
        st.write(actors_images.iloc[4,0])
        st.image(actors_images.iloc[4,1])
    with col_movie6:
        st.write(actors_images.iloc[5,0])
        st.image(actors_images.iloc[5,1])
    with col_movie7:
        st.write(actors_images.iloc[6,0])
        st.image(actors_images.iloc[6,1])

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # with open('cosine_similarity.pkl','rb') as f:
    #     cs = pkl.load(f)
    #
    # st.subheader('Recommended Movies')
    # movie_df
    #
    # movie_id = movie_df.index.values[0]
    #
    # scores = list(enumerate(cs[movie_id]))
    # sorted_scores = sorted(scores, key = lambda x: x[1], reverse = True)
    # sorted_scores = sorted_scores[1:]
    #
    # colr1,colr2,colr3,colr4,colr5 = st.columns(5)
    # space = '   '
    # with colr1:
    #
    #     movie_title = df[df.index == sorted_scores[0][0]]['name'].values[0]
    #     year_of_movie = df[df.index ==  sorted_scores[0][0]]['year'].values[0]
    #     st.write(movie_title,str(year_of_movie))
    #     st.image(df[df.name == movie_title]['images_links'].values[0])
    #     st.write(sorted_scores[0][1])
    # with colr2:
    #
    #     movie_title = df[df.index == sorted_scores[1][0]]['name'].values[0]                                             #Filme recomandate
    #     year_of_movie = df[df.index ==  sorted_scores[1][0]]['year'].values[0]
    #     st.write(movie_title,str(year_of_movie))
    #     st.image(df[df.name == movie_title]['images_links'].values[0])
    #     st.write(sorted_scores[1][1])
    # with colr3:
    #
    #     movie_title = df[df.index == sorted_scores[2][0]]['name'].values[0]
    #     year_of_movie = df[df.index ==  sorted_scores[2][0]]['year'].values[0]
    #     st.write(movie_title,str(year_of_movie))
    #     st.image(df[df.name == movie_title]['images_links'].values[0])
    #     st.write(sorted_scores[2][1])
    # with colr4:
    #
    #     movie_title = df[df.index == sorted_scores[3][0]]['name'].values[0]
    #     year_of_movie = df[df.index ==  sorted_scores[3][0]]['year'].values[0]
    #     st.write(movie_title,str(year_of_movie))
    #     st.image(df[df.name == movie_title]['images_links'].values[0])
    #     st.write(sorted_scores[3][1])
    # with colr5:
    #
    #     movie_title = df[df.index == sorted_scores[4][0]]['name'].values[0]
    #     year_of_movie = df[df.index ==  sorted_scores[4][0]]['year'].values[0]
    #     st.write(movie_title,str(year_of_movie))
    #     st.image(df[df.name == movie_title]['images_links'].values[0])
    #     st.write(sorted_scores[4][1])

# ------------------------------------------------------------------------------------------------------------

    # st.header('Sentiment')
    cols1,cols2 = st.columns([1,7])
    with cols1:
        st.header('Sentiment')

    with cols2:
        st.write(' ')
        st.write(' ')
        sentiment_checkbox = st.checkbox('')
    if sentiment_checkbox:

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

        link = movie_df['movie_link'].values[0]
        @st.cache(allow_output_mutation=True)
        def df_sentiment(link_movie):
            url = (
                link + "reviews/_ajax?ref_=undefined&paginationKey={}"
            )
            key = ""
            data = {"title": [], "review": [],"score":[]}

            while True:
                response = requests.get(url.format(key))
                soup = BeautifulSoup(response.content, "html.parser")
                # Find the pagination key
                pagination_key = soup.find("div", class_="load-more-data")                                  #scraper pentru movie reviews pe care le stocam in df_movie_sentiment
                if not pagination_key:
                    break

                # Update the `key` variable in-order to scrape more reviews
                key = pagination_key["data-key"]
                for title, review, score in zip(
                    soup.find_all(class_="title"), soup.find_all(class_="text show-more__control"),soup.find_all('span',class_="rating-other-user-rating")
                ):
                    data['score'].append(int(score.find_all('span')[0].text))
                    data["title"].append(title.get_text(strip=True))
                    data["review"].append(review.get_text())
            return pd.DataFrame(data)

        df_movie_sentiment = df_sentiment(link)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
        def sentiment_analyzer(text):
            score = SentimentIntensityAnalyzer().polarity_scores(text)
            pos = score['pos']
            neg = score['neg']
            if pos > neg:
              return ('positive')                                                   # functie pentru a returna coeficientii sentimentului: pos, neg, neu
            elif pos < neg:
              return ('negative')
            else:
              return ('neutral')

        df_movie_sentiment['sentiment'] = df_movie_sentiment['review'].apply(sentiment_analyzer)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

        positive_percentage = round(df_movie_sentiment.sentiment.value_counts()['positive']/df_movie_sentiment.sentiment.count(),2)
        avg_score_users = round(df_movie_sentiment.score.mean(),2)
        st.write(str(positive_percentage),'% of users have a Positive Sentiment')                    #calculeaza procentajul de review-uri pozitive si media notelor data de cei care au lasat review
        st.write(str(avg_score_users),'/10 score given by users who left a review')






    # with st.expander('Expander'):
    #     st.write('Expander')


# ----------------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------------
    atribut = st.selectbox('Alege un parametru:',['nota','metascore','votes_numerical','duration_numerical'])
                                                                                                                              # selectam un atribut si afisam dataframe-ul corespunzator
    df_atribut = pd.DataFrame(df[atribut].value_counts().reset_index().rename(columns={'index':atribut,atribut:'count'}))
    df_atribut
# ----------------------------------------------------------------------------------------------------------------------------
    try:
        percentile = df_atribut[df_atribut[atribut] <int(movie_df[atribut].values[0])]['count'].sum()/df_atribut['count'].sum()   # pe baza atributului ales calculam percentile
        percentile
    except:
        st.write(f'Movie does not have {atribut}')
# ----------------------------------------------------------------------------------------------------------------------------

    log_value = st.checkbox('Logarithm')   #parametru pentru functia 'plt.hist' de mai jos
# ---------------------------------------------------------------------------------------
    def get_ox_index(val,ox):
        for i in range(0,len(ox)-1):
            if ox[i] <= val <= ox[i+1]:
                index = i
        return index
# -----------------------------------------------------------------------------------------
    fig,ax = plt.subplots()
    n,bins,patches = ax.hist(x = df[atribut],bins=45,ec='white',log = log_value)

    try:
        a=get_ox_index(movie_df[atribut].values[0],list(bins))
        patches[a].set_fc('orange')                                                        # afisam graficul si coloram portocaliu bara corespunzatoare valorii atributului filmului selectat
        st.pyplot(fig)
    except:
        st.write(f'Can not plot the graph because Movie does not have {atribut}')
# ------------------------------------------------------------------------------------------

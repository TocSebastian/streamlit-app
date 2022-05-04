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
import pickle as pkl
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import imdb
import altair as alt
import ast
from wordcloud import WordCloud, STOPWORDS
import json
from streamlit_lottie import st_lottie

img = Image.open('logo.svg.png')

#make it look nice from the start
st.set_page_config(page_title = 'Movies IMDb',page_icon=img, layout='wide',initial_sidebar_state='collapsed',)

# specify the primary menu definition
menu_data = [
    {'icon': "bi bi-camera-reels-fill", 'label':"Movies Dataframe"},
    {'id':'Movie','icon':"bi bi-film",'label':"Movie"},
    {'id':'Analytics','icon':"bi bi-bar-chart-line",'label':"Analytics"}
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



if menu_id == 'Home':

    def load_lottie_url(url:str):
        r = requests.get(url)
        if r.status_code != 200:
            return  None
        return r.json()

    lottie_wave = load_lottie_url('https://assets3.lottiefiles.com/packages/lf20_dpohsucu.json')

    lottie_graphs = load_lottie_url('https://assets7.lottiefiles.com/packages/lf20_sr19osee.json')

    lottie_movie = load_lottie_url('https://assets7.lottiefiles.com/packages/lf20_cbrbre30.json')




    hp1, hp2 =st.columns(2)

    with hp1:

        st.markdown('##')
        html_hello = f"""
        <style>
        p.a {{
          font: bold 20px Courier;
        }}
        </style>
        <p class="a">Hi, i am Sebastian 👋  </p>
        """
        st.markdown(html_hello, unsafe_allow_html = True)

        # st.write('Avem aici un GIF',"![Your Awsome GIF](https://www.pngfind.com/pngs/m/28-288401_waving-hand-emoji-svg-png-download-emoji-hand.png)")
        # st.write('Avem aici un GIF',"![Your Awsome GIF](https://c.tenor.com/z2xJqhCpneIAAAAM/wave-hand.gif)")

        st.header('An aspiring Data Scientist from Romania')
        st.write('I am passionate about Data Science and Movies and this project is the product of that.')

        st_lottie(lottie_movie,speed =1,quality = 'low',height = 300, width = 700, key = 'Movie')

    with hp2:
        st_lottie(lottie_graphs,speed =1,quality = 'low',height = 300, width = 700, key = 'Graph')


    # st.markdown('##')
    st.header('📬 Connect with me ')

    ab1,ab2,ab3 = st.columns([1,1,80])



    with ab1:
        st.image('linkedin.png', width = 25)
        st.image('github.png', width = 40)

    with ab3:
        st.write("[LinkedIn](https://www.linkedin.com/in/toc-sebastian-b15193217/)")
        st.write("[GitHub](https://github.com/TocSebastian)")


# ------------------------------------------------------------------------------
elif menu_id == 'Movies Dataframe':
    st.title("Movies by Rating")

    df = pd.read_pickle('df8k.pkl')                                             # incarcare dataframe precum si 2 liste cu actori si directori de filmi fara duplicate
    df = df.set_index('name').reset_index()
    df['metascore'] = pd.to_numeric(df['metascore'])
    list_set_actors = pd.read_pickle('list_set_actors.pkl')
    list_set_directors = pd.read_pickle('list_set_directors.pkl')
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
        metascore = st.sidebar.select_slider('Metascore',metascore_colors,'green')          #
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
    df = pd.read_pickle('df8k.pkl')
    movie_links_images = pd.read_pickle('all_movies_images_links.pkl')          # pagina noua
    df["images_links"] = movie_links_images                                     # incarcam datafraem-ul, lista cu linkuri ale imaginilor si combinam in acelasi dataframe
    movies = pd.read_pickle('list_movie_names_years.pkl')                       #incarcam lista cu nume_film + an pentru a putea cauta filmele corespunzator
    movies_with_links = pd.read_pickle('Top8kMovies.pkl')
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
        st.markdown('#')                                                        # aici afisam nota si numarul de voturi folosind functiile html de mai devreme si deasemenea afisam si metascore-ul
        st.markdown('#')                                                        # intr-un mod highlithed in fucntie de valoare folosind functia de mai sus 'color_metascore' precum si f string
        st.markdown('#')
        st.markdown('#')

        try:
            t = f"""<div><span class='highlight {color_metascore(int(movie_df['metascore'].values[0]))}'> <span class='bold'>{movie_df['metascore'].values[0]}</span> </span><span class='bold'>Metascore</span></div>"""
        except:
            t = f"""<div><span class='highlight grey'><span class='bold'>No</span></span><span class='bold'> Metascore</span></div> """
        st.markdown(t, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # movie_df  comentariu ----------------------------

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
    # actors_images comentariu ------------------------
#     x.img['srcset'].split(',')

    st.markdown('#')
    st.subheader('Actors')
    not_found = 'https://t3.ftcdn.net/jpg/03/35/13/14/360_F_335131435_DrHIQjlOKlu3GCXtpFkIG1v0cGgM9vJC.jpg'

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
        try:
            st.write(actors_images.iloc[3,0])
            st.image(actors_images.iloc[3,1])
        except:
            st.write('Not found')
            st.image(not_found,width=180)
    with col_movie5:

        try:
            st.write(actors_images.iloc[4,0])
            st.image(actors_images.iloc[4,1])
        except:
            st.write('Not found')
            st.image(not_found,width=180)
    with col_movie6:
        try:
            st.write(actors_images.iloc[5,0])
            st.image(actors_images.iloc[5,1])
        except:
            st.write('Not found')
            st.image(not_found,width=180)
    with col_movie7:
        try:
            st.write(actors_images.iloc[6,0])
            st.image(actors_images.iloc[6,1])
        except:
            st.write('Not found')
            st.image(not_found,width=180)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def cs_pickle():
        with open('cs1.pkl','rb') as f:
            cs1pkl = pkl.load(f)
        with open('cs2.pkl','rb') as f:
            cs2pkl = pkl.load(f)
        with open('cs3.pkl','rb') as f:
            cs3pkl = pkl.load(f)
        with open('cs4.pkl','rb') as f:
            cs4pkl = pkl.load(f)
        with open('cs5.pkl','rb') as f:
            cs5pkl = pkl.load(f)
        with open('cs6.pkl','rb') as f:
            cs6pkl = pkl.load(f)

        cs = np.hstack([cs1pkl,cs2pkl,cs3pkl,cs4pkl,cs5pkl,cs6pkl])
        return cs

    cs = cs_pickle()

    st.markdown('#')
    st.subheader('Recommended Movies')
    # movie_df comentariu -----------------------------

    movie_id = movie_df.index.values[0]

    scores = list(enumerate(cs[movie_id]))
    sorted_scores = sorted(scores, key = lambda x: x[1], reverse = True)
    sorted_scores = sorted_scores[1:]

    colr1,colr2,colr3,colr4,colr5 = st.columns(5)
    space = '   '
    with colr1:

        movie_title = df[df.index == sorted_scores[0][0]]['name'].values[0]
        year_of_movie = df[df.index ==  sorted_scores[0][0]]['year'].values[0]
        st.write(movie_title,str(year_of_movie))
        st.image(df[df.name == movie_title]['images_links'].values[0])
        t =  f"""  <span class='bold'>{round(sorted_scores[0][1],2)*100} </span> % Similarity """
        st.markdown(t, unsafe_allow_html=True)

    with colr2:

        movie_title = df[df.index == sorted_scores[1][0]]['name'].values[0]                                             #Filme recomandate
        year_of_movie = df[df.index ==  sorted_scores[1][0]]['year'].values[0]
        st.write(movie_title,str(year_of_movie))
        st.image(df[df.name == movie_title]['images_links'].values[0])
        t =  f"""  <span class='bold'>{round(sorted_scores[1][1],2)*100} </span> % Similarity """
        st.markdown(t, unsafe_allow_html=True)
    with colr3:

        movie_title = df[df.index == sorted_scores[2][0]]['name'].values[0]
        year_of_movie = df[df.index ==  sorted_scores[2][0]]['year'].values[0]
        st.write(movie_title,str(year_of_movie))
        st.image(df[df.name == movie_title]['images_links'].values[0])
        t =  f"""  <span class='bold'>{round(sorted_scores[2][1],2)*100} </span> % Similarity """
        st.markdown(t, unsafe_allow_html=True)
    with colr4:

        movie_title = df[df.index == sorted_scores[3][0]]['name'].values[0]
        year_of_movie = df[df.index ==  sorted_scores[3][0]]['year'].values[0]
        st.write(movie_title,str(year_of_movie))
        st.image(df[df.name == movie_title]['images_links'].values[0])
        t =  f"""  <span class='bold'>{round(sorted_scores[3][1],2)*100} </span> % Similarity """
        st.markdown(t, unsafe_allow_html=True)
    with colr5:

        movie_title = df[df.index == sorted_scores[4][0]]['name'].values[0]
        year_of_movie = df[df.index ==  sorted_scores[4][0]]['year'].values[0]
        st.write(movie_title,str(year_of_movie))
        st.image(df[df.name == movie_title]['images_links'].values[0])
        t =  f"""  <span class='bold'>{round(sorted_scores[4][1],2)*100} </span> % Similarity """
        st.markdown(t, unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------------------

    # st.header('Sentiment')
    with st.expander('Sentiment'):
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
    with st.expander('Parameter Analysis'):
        st.markdown('#')
        atribut = st.selectbox('Alege un parametru:',['nota','metascore','votes_numerical','duration_numerical'])
                                                                                                                                  # selectam un atribut si afisam dataframe-ul corespunzator
        df_atribut = pd.DataFrame(df[atribut].value_counts().reset_index().rename(columns={'index':atribut,atribut:'count'}))
        # df_atribut comentariu ---------------------------

    # ----------------------------------------------------------------------------------------------------------------------------
        try:
            # st.markdown('##')
            percentile = df_atribut[df_atribut[atribut] < float(movie_df[atribut].values[0])]['count'].sum()/df_atribut['count'].sum()   # pe baza atributului ales calculam percentile
            rank = percentile * 8332
            t =  f""" The Movie <span class='bold'>{movie} </span> is in the <span class='bold'> {str(round(percentile,5))} </span>percentile for the attribute <span class='bold'>{atribut}</span> """
            # t =  f""" The Movie <span class='bold'>{movie} </span> is in the <span class='bold'> {percentile} </span> for the attribute <span class='bold'>{atribut}</span> """
            st.markdown(t, unsafe_allow_html=True)
            t =  f""" The Movie <span class='bold'>{movie} </span> has the Rank <span class='bold'> {round(rank)} </span>/8331 for the attribute <span class='bold'>{atribut}</span> """
            # t =  f""" The Movie <span class='bold'>{movie} </span> is in the <span class='bold'> {percentile} </span> for the attribute <span class='bold'>{atribut}</span> """
            st.markdown(t, unsafe_allow_html=True)



        except:
            st.write(f'Movie does not have {atribut}')
    # ----------------------------------------------------------------------------------------------------------------------------
        st.markdown("##")
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
    with st.expander('Budget and Box Office'):
        moviesDB = imdb.IMDb()
        movies = moviesDB.search_movie(movie_df['name'].values[0])
        # movie_df['name'].values[0]
        title = movies[0]['title']
        year = movies[0]['year']


        movie_id = movies[0].getID()
        movie = moviesDB.get_movie(movie_id)
        try:
            budget = int(movie['box office']['Budget'].split(' ')[0][1:].replace(',',''))
            box_office = int(movie['box office']['Cumulative Worldwide Gross'].split(' ')[0][1:].replace(',',''))

            # budget
            # box_office

            df_budget_box_office = pd.read_pickle('df_budget_box_office.pkl')
            budget_avg = df_budget_box_office['Production Budget'].mean()
            box_office_avg = df_budget_box_office['Worldwide Gross'].mean()

            def millions(x):
                return round((x/10**6),2)

            def dollar_converter(x):
                if x//10**9 >= 1:
                    return str(round(x/10**9,2)) + ' Billion Dollars'
                elif x//10**6 >= 1:
                    return str(round(x/10**6,2)) + ' Million Dollars'
                elif x//10**3 >= 1:
                    return str(round(x/10**3,2)) + ' Thounsands Dollars'
                else:
                    return str(x) + ' Dollars'


            labels = [movie_df['name'].values[0],'Average']
            budget_l = [millions(budget),millions(budget_avg)]
            box_office_l = [millions(box_office-budget),millions(box_office_avg-budget_avg)]

            width = 0.4     # the width of the bars: can also be len(x) sequence
# ------------------------------------------------------------------------------
            fig, ax = plt.subplots()

            ax.bar(labels, budget_l, width, label='Budget')
            ax.bar(labels, box_office_l, width,  bottom=budget_l,
                   label='Box Office')                                        # another way to plot budget and box office using matplotlib

            ax.set_ylabel('Million of Dollars')
            ax.set_title('Profits')
            ax.legend()
# ------------------------------------------------------------------------------
            # plt.show()
            # budget_avg
            # box_office_avg
            # budget_l
            # box_office_l

            dict_nou = {'Budget': budget_l,'Box Office':box_office_l,'Type':[movie_df['name'].values[0],'Average Moivie']}
            df_now = pd.DataFrame(dict_nou)
            df_now.set_index('Type',inplace = True)

            cfig1, cfig2, cfig3 = st.columns([10,1,8])


            with cfig1:
                st.pyplot(fig)
            with cfig3:

                movie_budget = f"""<div><span class='bold'>Production Budget: {dollar_converter(budget)} </span></div>"""
                movie_box_office = f"""<div><span class='bold'>Worldwide Gross: {dollar_converter(box_office)} </span></div>"""

                movie_budget_avg = f"""<div><span class='bold'>Production Budget: {dollar_converter(budget_avg)} </span></div>"""
                movie_box_office_avg = f"""<div><span class='bold'>Worldwide Gross: {dollar_converter(box_office_avg)} </span></div>"""

                st.header(movie_df['name'].values[0])
                st.markdown(movie_budget, unsafe_allow_html=True)
                st.markdown(movie_box_office, unsafe_allow_html=True)

                st.markdown('###')

                st.header('Average values for a Movie')
                st.markdown(movie_budget_avg, unsafe_allow_html=True)
                st.markdown(movie_box_office_avg, unsafe_allow_html=True)
        except:
            st.header('No Budget and Box Office data for the movie ' + movie_df['name'].values[0])
            st.image('not_possible.jpg')
        # st.video('https://www.youtube.com/watch?v=coY2IA-oBvw&ab_channel=FuckyouGoogle',format="video/mp4", start_time=0)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    with st.expander('Votes Distribution'):
        vd1,vd2,vd3 = st.columns([10,1,6])
        with vd1:
            url = movie_df['movie_link'].values[0] + 'ratings'
            df_movie_ratings = pd.DataFrame()
            page = requests.get(url).text
            doc = BeautifulSoup(page, 'html.parser')
            ratings = doc.find('table').find_all('tr')[1:]
            i = 10
            for rating in ratings:
                df_movie_ratings.at[0,i] = int(rating.find(class_="leftAligned").text.replace(',',''))
                i=i-1

            df_movie_ratings = df_movie_ratings.rename(index = {0:'Votes'})
            row = df_movie_ratings.iloc[0]
            fig = row.plot(kind='bar')
            plt.show()
            st.write(' ')
            st.bar_chart(row,height=450)

            # st.pyplot(fig=plt)
        with vd3:
            def number_converter(x):
                if x//10**6 >= 1:
                    return str(round(x/10**6,2)) + ' M Votes'
                elif x//10**3 >= 1:
                    return str(round(x/10**3,2)) + ' K Votes'
                else:
                    return str(x) + ' Votes'
            st.markdown('##')
            for i in range(10,0,-1):
                t=f"""<div>Rating {i}<span class='bold'>: {number_converter(df_movie_ratings[i].values[0])} </span></div>"""
                st.markdown(t,unsafe_allow_html=True)
                st.write(' ')




elif menu_id == "Analytics":

    st.header('Analytics')
    df = pd.read_pickle('df_movies_budgets_actors_2022.pkl')

# ---------------------------------------------------------------------------------------------------------------------

    with st.expander('Directors'):



        df['Counts'] = df.groupby('director')['director'].transform('count')




        df_directors = df[df.groupby('director')['director'].transform('size') > 4]

        num_movies = st.slider('Number of movies', 5, int(df_directors['director'].value_counts().max()), 10)

        df_directors = df[df.groupby('director')['director'].transform('size') >= num_movies]

        col_director1, col_director2 = st.columns(2)

        with col_director1:

            director_param = st.selectbox('Select parameter:',['Worldwide Gross','nota','metascore','votes_numerical','duration_numerical'])

        with col_director2:
            gross_func = st.selectbox('Select function:',['mean','max','min'])

        # df_directors

        df_top_directors = df_directors.groupby(['director','Counts']).agg({director_param:gross_func})

        # df_top_directors

        def round_column(x):
            return round(x,2)


        df_top_directors = df_top_directors[director_param].apply(round_column)
        df_top_directors

        top_directors = pd.DataFrame(df_top_directors).sort_values('Counts',ascending = False)
        top_directors.reset_index(inplace = True)


        top_directors['Director/Count'] = top_directors['director']+ ' (' + top_directors['Counts'].astype(str) + ')'
        # top_directors


        # fig = plt.figure(figsize = (8,4))
        # sns.barplot(x =director_param, y = 'Director/Count', data = top_directors.iloc[:20])
#-------------------------------------------------------------------------------
        bars = alt.Chart(top_directors).transform_window(rank='rank('+ director_param +')',sort=[alt.SortField(director_param, order='descending')]).transform_filter(alt.datum.rank <= 20).mark_bar().encode(
        x = director_param,
        y =  alt.Y('Director/Count', sort = '-x'),
        tooltip = ['director',director_param]
        )

        text = bars.mark_text(
        align='left',
        baseline='middle',                                                          # altair plot
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
        ).encode(
            text=director_param
        )

        st.markdown('#')
        st.altair_chart(bars,use_container_width=True)  #(bars + text).properties(height=900) in loc de bars

# ------------------------------------------------------------------------------
    with st.expander('Budget and Worldwide Gross'):



        df = pd.read_pickle('df_movies_budgets_actors.pkl')
        df_ratings = pd.read_pickle('df8k.pkl')
        # df_ratings
        df_new = pd.merge(df_ratings[['name', 'nota']], df,  left_on = 'name', right_on = 'Name')
        df_new = df_new.sort_values('nota', ascending = False).drop_duplicates('name', keep = 'first')
        # df1.sort_values('Count').drop_duplicates('Name', keep='last')
        # df_new.shape

        num_of_movies = st.slider('Number of Movies displayed:',100,df_new.shape[0],100,50)

        base = alt.Chart(df_new.iloc[:num_of_movies]).mark_circle().encode(
        x = 'Production Budget', y = 'Worldwide Gross', color = 'nota', size = 'Production Budget', tooltip = ['name','Production Budget','Worldwide Gross','nota'])

        # chart = alt.layer(
        # base.mark_rule().encode(alt.Y('Worldwide Gross', title='Price',
        #                             scale=alt.Scale(zero=False)), alt.Y2('Worldwide Gross'))).interactive()
    # base.mark_bar().encode(alt.Y('Open:Q'), alt.Y2('Close:Q')),).interactive()
        st.markdown('#')
        st.altair_chart(base, use_container_width = True )


        # df = pd.DataFrame(
        #  np.random.randn(200, 3),
        #  columns=['a', 'b', 'c'])
        #
        # c = alt.Chart(df).mark_circle().encode(
        #      x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
        #
        # st.altair_chart(c, use_container_width=True)
# --------------------------------------------------------------------------------------------------------------------------------
    with st.expander('Worldwide Gross by Genres'):
        df = pd.read_pickle('df_movies_budgets_actors_2022.pkl')


        def get_genres(x):
            try:
                return x[0]
            except:
                return None

        df['principal_genre'] = df['genres'].apply(get_genres)
        df_genres = pd.DataFrame(df.groupby(['principal_genre','year'])['Worldwide Gross'].sum()).reset_index()
        top_10_genres = df.groupby('principal_genre',as_index = False)['Worldwide Gross'].sum().sort_values('Worldwide Gross', ascending = False)[:10]['principal_genre'].to_list()
        df_genres_to_plot = df_genres[df_genres['principal_genre'].isin(top_10_genres)]
        top_genres = df.groupby('principal_genre',as_index = False)['Worldwide Gross'].sum().sort_values('Worldwide Gross', ascending = False)['principal_genre'].to_list()



        genres_choice =  st.checkbox('Genres')
        if genres_choice:

            col_gen_1, col_gen_2 = st.columns([3,1])
            with col_gen_1:

                genres_choosed = st.multiselect("Alege gen:",top_genres,'Action')
                df_genres_to_plot = df_genres[df_genres['principal_genre'].isin(genres_choosed)]

            with col_gen_2:
                st.markdown('#')

                select_all = st.checkbox('Select all')
                if select_all:
                    df_genres_to_plot = df_genres[df_genres['principal_genre'].isin(top_genres)]



        fig = plt.figure(figsize=(12,4))
        sns.lineplot(x = 'year', y = 'Worldwide Gross', hue = 'principal_genre', data = df_genres_to_plot)

        # st.pyplot(fig)


        base = alt.Chart(df_genres_to_plot).mark_line().encode(
        x='year',
        y='Worldwide Gross',
        color='principal_genre'
        #strokeDash='principal_genre',
        )
        st.markdown('#')
        st.altair_chart(base, use_container_width = True)

    with st.expander('Genres Frequency'):

        movie_genres = df['genres'].to_list()



        genres_list = []

        for x in movie_genres:
            try:
                for i in x:
                    genres_list.append(i)
            except:
                pass


        genres_unique = list(set(genres_list))


        dict_frequency = {}

        for x in genres_unique:
            dict_frequency[x] = genres_list.count(x)


        tone = 100 # define the color of the words
        f, ax = plt.subplots(figsize=(14, 6))
        wordcloud = WordCloud(width=550,height=300, background_color='white',
                              max_words=1628,relative_scaling=0.7,
                              normalize_plurals=False)
        c = wordcloud.generate_from_frequencies(dict_frequency)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.markdown('#')
        st.pyplot()

# --------------------------------------------------------------------------------------------

    with st.expander('Heatmap'):

        df = pd.read_pickle('df_grouped_months_years.pkl')

        a = df.pivot('month_number','year','Worldwide Gross')

        fig = plt.figure(figsize=(12,4))
        ax = sns.heatmap(a, cmap = 'Blues', linewidths=.5)

        st.pyplot(fig)

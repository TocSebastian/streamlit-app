import streamlit as st
import pandas as pd
from load_css import local_css

df = pd.read_pickle('df8k.pkl')
movie_links_images = pd.read_pickle('all_movies_images_links.pkl')
df["images_links"] = movie_links_images
movies = pd.read_pickle('list_movie_names_years.pkl')



# st.set_page_config(layout="wide")



movie_year = st.selectbox("Alege un film",movies)

movie_splitted= movie_year.rsplit(' ',1)
movie = movie_splitted[0]
year_movie = int(movie_splitted[1])

st.title(movie)

movie_df = df[(df.name == movie) & (df.year == year_movie)]

yrd = movie_df[['year','R','duration']]

year = yrd['year'].values[0]
rated = yrd['R'].values[0]
duration = yrd['duration'].values[0]

yrd_output = str(str(year) + ' ' + str(rated) + ' ' + str(duration))

st.session_state.emoji = "⭐"

html_yrd = f"""
<style>
p.a {{
  font: bold 20px Courier;
}}
</style>
<p class="a">{year}  {rated}  {duration} </p>
"""

html_rating = f"""
<style>
p.a {{
  font: bold 20px Courier;
}}
</style>
<p class="a">{st.session_state.emoji} {movie_df['nota'].values[0]}/10 </p>
"""


html_votes = f"""
<style>
p.a {{
  font: bold 20px Courier;
}}
</style>
<p class="a">{movie_df['votes'].values[0]} votes</p>
"""
def f_str(x,sep = ' '):
    l=''
    for i in x:
        l = l + i + sep
    return l


local_css("style.css")
def color_metascore(x):
    if x > 60:
        color = 'green'
    elif x > 40:
        color = 'yellow'
    else:
        color = 'red'
    return color



col1,col2,col3 = st.columns([1,3,1])

with col1:
    st.markdown(html_yrd, unsafe_allow_html=True)
    st.image(movie_df['images_links'].values[0])

with col2:
    st.header('Details')
    try:
        st.write('Genres:', f_str(movie_df['genres'].values[0]))
    except:
        st.write('Genres:', None )
    st.write('Description')
    st.write(movie_df['description'].values[0])
    st.write('Actors:', f_str(movie_df['actors'].values[0],', ')[:-2])
    st.write('Director:', movie_df['director'].values[0])
    st.write('Writers:', f_str(movie_df['actors'].values[0],', ')[:-2])
    try:
        st.write('Tagline:', movie_df['tagline'].values[0])
    except:
        st.write('Tagline:', None )



with col3:

    st.markdown(html_rating, unsafe_allow_html = True)
    st.markdown(html_votes, unsafe_allow_html = True)
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')

    try:
        t = f"""<div><span class='highlight {color_metascore(int(movie_df['metascore'].values[0]))}'> <span class='bold'>{movie_df['metascore'].values[0]}</span> </span><span class='bold'>Metascore</span></div>"""
    except:
        t = f"""<div><span class='highlight grey'><span class='bold'>No</span></span><span class='bold'> Metascore</span></div> """
    st.markdown(t, unsafe_allow_html=True)


df.metascore.value_counts()


movie_df

# st.markdown("⭐")
# st.write(year,rated,duration)

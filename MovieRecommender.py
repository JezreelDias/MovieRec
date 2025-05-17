import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_data():
    df=pd.read_csv("MovieRecommender/tmdb_5000_movies.csv")
    return df

def calculate_weighted_ratings(df):
    C=df["vote_average"].mean()
    m=df["vote_count"].quantile(0.9)
    q_movies=df.copy().loc[df["vote_count"]>m]
    q_movies["score"] = q_movies.apply(weighted_rating,axis=1,m=m,C=C)
    q_movies=q_movies.sort_values("score",ascending=False).head(10)
    return q_movies

def weighted_rating(x,m,C):
    v=x["vote_count"]
    R=x["vote_average"]
    return ((v/(v+m))*R)+((m/(v+m))*C)

def get_top10_movies_by_score(q_movies):
    st.header("Top 10 Movies by Weighted Score 🚀")
    wr = q_movies[["title","score","id"]].head(10)
    st.dataframe(wr)

def get_top10_movies_by_popularity(df):
    st.header("Top 10 movies based on imbd popularity rating score🔥")
    pop = q_movies[["title","popularity","id"]].sort_values('popularity',ascending=False).head(10)
    st.dataframe(pop)

def get_personal_reccomendation(title,df):
    tf_idf_matrix,sim,indices=create_tfidf_cosine_matrix(df)
    idx=indices[title]
    sim_scores=list(enumerate(sim[idx]))
    sort_sim_scores=sorted(sim_scores, key=lambda x:x[1],reverse=True)
    sort_sim_scores=sort_sim_scores[1:11]
    movie_indices=[]
    for i in sort_sim_scores:
        movie_indices.append(i[0])    
    return df["title"].iloc[movie_indices]

def create_tfidf_cosine_matrix(df):
    vectorizer=TfidfVectorizer(stop_words="english")
    df["overview"]=df["overview"].fillna("")
    tfidf_matrix=vectorizer.fit_transform(df['overview'])
    sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
    indices=pd.Series(df.index,index=df["title"]).drop_duplicates()
    return tfidf_matrix,sim,indices

st.title("Movie Recommender System 🎥")
df=read_data()
q_movies = calculate_weighted_ratings(df.copy())

menu=st.sidebar.radio("Choose a Recommender Type:",
                          ["Top 10 Movies by Weighted Score",
                          "Top 10 Popular Movies",
                          "Get Personilized Recommendation"])

if menu=="Top 10 Movies by Weighted Score":
    get_top10_movies_by_score(q_movies.copy())
elif menu=="Top 10 Popular Movies":
    get_top10_movies_by_popularity(df.copy())
elif menu=="Get Personilized Recommendation":
    st.header("Get Personalized Recommendation")
    st.subheader("Top 10 Movies Based on TFIDF and Cosine Similarity")
    selected_movies = [
      "Harry Potter and the Chamber of Secrets",
      "Harry Potter and the Philosopher's Stone",
      "The Hobbit: The Desolation of Smaug", "Avatar", "Spider-Man 3",
      "Avengers: Age of Ultron", "Iron Man", "Iron Man 2",
      "X-Men: The Last Stand", "Star Trek Beyond", "The Fast and the Furious",
      "How to Train Your Dragon", "Mission: Impossible - Rogue Nation",
      "Minions"
  ]
    selected_movie=st.selectbox("Select a Movie🍿",selected_movies)

    if st.button("Get Recommendation 💯"):
        recommendations=get_personal_reccomendation(selected_movie,df.copy())
        st.dataframe(recommendations)
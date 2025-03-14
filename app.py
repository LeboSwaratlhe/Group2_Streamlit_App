import streamlit as st
import dill
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as CS
from surprise import SVD, Reader, Dataset
import emoji
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# Load pickled data
@st.cache_resource
def load_pickled_data():
    try:
        with open("function_dict4.pkl", "rb") as f:
            pickled_data = dill.load(f)
        return pickled_data
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Load pre-trained SVD model
@st.cache_resource
def load_ncf_model():
    try:
        ncf_model = load_model("ncf_model.h5", safe_mode=False)
        return ncf_model
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()

# Streamlit UI
st.set_page_config(page_title="Anime Recommendation System", page_icon="🎬")

# Use emoji package for shortcodes
def emojize(text):
    return emoji.emojize(text)


st.title(emojize(":clapper: Anime Recommendation System"))

# Add image below the title
st.image("https://storage.googleapis.com/kaggle-datasets-images/571/1094/c633ae058ddaa59f43649caac1748cf4/dataset-card.png", caption="Anime Recommendation System")

# Create a team tab
tab1, tab2 = st.tabs(["Recommendation", "Team"])

with tab1:
    # Load data
    pickled_data = load_pickled_data()
    df = pickled_data['df1']
    pca_df = pickled_data['df2']

    def get_encoders():
        with open('anime_encoder.pkl', 'rb') as file:
            anime_encoder = pickle.load(file)

        with open('user_encoder.pkl', 'rb') as file2:
            user_encoder = pickle.load(file2)

        return user_encoder, anime_encoder

    train = pd.read_csv('updated_train.csv')
    valid_user_ids = set(train['user_id'])


    # Recommendation method selection
    model = st.radio(
        emojize(":mag_right: Select Recommendation Method"),
        ['Content-Based (PCA)',
         'Collaborative-Based (Rating Predictor - NCF)',
         'Collaborative-Based (User Recommendations)',
         'Hybrid (PCA + NCF)'],
        index=0
    )

    # User Input
    anime_titles = df['name'].unique()
    

    # PCA-based recommendation function
    def recommend_anime_pca30(input_anime, df, pca_df, top_n=10):
        if input_anime not in df['name'].to_numpy():
            return None

        title_index = df[df['name'] == input_anime].index[0]
        input_pca_vector = pca_df.iloc[title_index, :-1].to_numpy().reshape(1, -1)

        similarities = CS(input_pca_vector, pca_df.iloc[:, :-1])
        similarities = similarities.flatten()

        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        recommendations = list(df.iloc[similar_indices]['name'])

        return recommendations, similarities[similar_indices]

    # Collaborative-based rating predictor function
    def get_predicted_rating(user_id, input_anime, ncf_model):
        user_encoder, anime_encoder = get_encoders()
        user_id_encoded = user_encoder.transform([user_id])[0]
        anime_id_encoded = anime_encoder.transform([input_anime])[0]
        
        pred = ncf_model.predict([np.array([[user_id_encoded]]), np.array([[anime_id_encoded]])])[0][0]

        pred = np.clip(pred, 1, 10)

        return pred

    # Collaborative-based recommendation function
    def recommend_anime_for_user(user_id, ncf_model, df, pca_df, top_n=10, alpha=0.0):

        udf = train[train['user_id']==user_id].sort_values(by='rating', ascending=False).head(10)

        if len(udf) < 10:
            y_pred = 0
            i = -1
            user_encoder, anime_encoder = get_encoders()
            while y_pred < 8:
                i+=1
                user_id_encoded = user_encoder.transform([user_id])[0]
                anime_id_encoded = anime_encoder.transform([train['anime_id'][i]])[0]
        
                y_pred = ncf_model.predict([np.array([[user_id_encoded]]), np.array([[anime_id_encoded]])])[0][0]

            recommendations, _ = recommend_anime_pca30(df[df['anime_id']==train['anime_id'][i]]['name'].to_numpy()[0], df, pca_df, 10)

            return recommendations
        
        else:
            return df[df['anime_id'].isin(udf['anime_id'])]['name'].tolist()

            
    # Hybrid recommendation function
    def recommend_anime_hybrid(input_anime, user_id, ncf_model, df, pca_df, top_n=10, alpha=0.5):
        pca_recs, pca_scores = recommend_anime_pca30(input_anime, df, pca_df, top_n=50)
        
        if pca_recs is None:
            return None

        ncf_predictions = {}
        ncf_scaled_preds = {}
        ncf_min, ncf_max = 1, 10

        user_encoder, anime_encoder = get_encoders()

        for anime in pca_recs:
            anime_id = df.loc[df['name'] == anime, 'anime_id'].iloc[0]
            user_id_encoded = user_encoder.transform([user_id])[0]
            anime_id_encoded = anime_encoder.transform([anime_id])[0]
        
            ncf_pred = ncf_model.predict([np.array([[user_id_encoded]]), np.array([[anime_id_encoded]])])[0][0]
            ncf_pred = np.clip(ncf_pred, 1, 10)
            ncf_scaled = round((ncf_pred - ncf_min) / (ncf_max - ncf_min), 4)
            ncf_predictions[anime] = round(ncf_pred, 2)
            ncf_scaled_preds[anime] = ncf_scaled

        hybrid_scores = {anime: round(alpha * ncf_scaled_preds[anime] + (1 - alpha) * pca_score, 2)
                         for anime, pca_score in zip(pca_recs, pca_scores)}

        sorted_anime = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract recommended anime, their hybrid scores and predicted ratings
        recommended_anime = [(anime, score, ncf_predictions[anime]) for anime, score in sorted_anime[:top_n]]
        
        return recommended_anime

    # Streamlit UI for Content-Based (PCA)
    if model == "Content-Based (PCA)":
        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        
        if st.button(emojize(":robot_face: Get Content-Based Recommendations")):
            with st.spinner('Generating recommendations...'):
                recommendations, _ = recommend_anime_pca30(anime_title, df, pca_df, top_n=10)
            if recommendations is None:
                st.write(":no_entry_sign: Anime not found.")
            else:
                st.write(":sparkles: Content-Based Recommendations:")
                for i, anime in enumerate(recommendations, 1):
                    st.write(f"{i}. {anime} :tv:")

    # Streamlit UI for Collaborative Filtering (Rating Predictor SVD)
    elif model == "Collaborative-Based (Rating Predictor - NCF)":

        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        #user_id = st.number_input(emojize(":id: Enter User ID:"), min_value=1, step=1)
        user_id_input = st.text_input(emojize(":id: Enter User ID:"))

        # Validate input
        if user_id_input:
            try:
                user_id = int(user_id_input)
                if user_id in valid_user_ids:
                    st.success(f"User ID is Valid: {user_id}")
                    if st.button(emojize(":robot_face: Predict Rating")):
                        with st.spinner('Generating predicted rating...'):
                            ncf_model = load_ncf_model()
                            # Get the anime_id for the selected title
                            matching_anime = df.loc[df['name'] == anime_title, 'anime_id']
                            if matching_anime.empty:
                                st.write(":no_entry_sign: Anime not found in the dataset.")
                            else:
                                anime_id = matching_anime.iloc[0]

                                # Predict rating for the selected anime
                                predicted_rating = get_predicted_rating(user_id, anime_id, ncf_model)
                                st.write(f"📊 **Predicted Rating for {anime_title}:** ⭐ {predicted_rating:.2f}")
                else:
                    st.error("Invalid User ID! Please enter a valid user ID.")
            except ValueError:
                st.error("Please enter a valid numeric User ID.")
                

    # Streamlit UI for Collaborative Filtering (User Recommendations)
    elif model == "Collaborative-Based (User Recommendations)":
        user_id_input = st.number_input(emojize(":id: Enter User ID:"), min_value=1, step=1)

        try:
            user_id = int(user_id_input)
            if user_id in valid_user_ids:
                st.success(f"User ID is Valid: {user_id}")

                if st.button(emojize(":robot_face: Get User-Based Recommendations")):
                    with st.spinner(f'Generating recommendations for user {user_id}...'):
                        ncf_model = load_ncf_model()
                        recommendations = recommend_anime_for_user(user_id, ncf_model, df, pca_df, top_n=10, alpha=0.0)
                    
                    if recommendations is None:
                        st.write(":no_entry_sign: Anime not found.")
                    else:
                        st.write(f":sparkles: Recommendations For User {user_id}:")
                        for i, anime in enumerate(recommendations, 1):
                            st.write(f"{i}. {anime} :tv:")

            else:
                st.error("Invalid User ID! Please enter a valid user ID.")
        except ValueError:
            st.error("Please enter a valid numeric User ID.")

    # Streamlit UI for Hybrid Model
    elif model == "Hybrid (PCA + NCF)":
        
        anime_title = st.selectbox(emojize(":film_frames: Select an anime title:"), anime_titles)
        user_id_input = st.number_input(emojize(":id: Enter User ID:"), min_value=1, step=1)
        alpha = st.slider("Weight for Hybrid (PCA + SVD)", 0.0, 1.0, 0.5, step=0.1)
        
        try:
            user_id = int(user_id_input)
            if user_id in valid_user_ids:
                st.success(f"User ID is Valid: {user_id}")

                if st.button(emojize(":robot_face: Get Hybrid Recommendations")):
                    with st.spinner('Generating recommendations...'):
                        ncf_model = load_ncf_model()
                        recommendations = recommend_anime_hybrid(anime_title, user_id, ncf_model, df, pca_df, top_n=10, alpha=alpha)
                    if recommendations is None:
                        st.write(":no_entry_sign: No recommendations found.")
                    else:
                        st.write(":sparkles: Hybrid Recommendations:")
                        for i, (anime, score, rating) in enumerate(recommendations, 1):
                            st.write(f"{i}. **{anime}** :star2: (Hybrid Score: {score:.2f}) :star: (Predicted Rating: {round(rating, 2)})")

            else:
                st.error("Invalid User ID! Please enter a valid user ID.")
        except ValueError:
            st.error("Please enter a valid numeric User ID.")    

with tab2:
    st.write("### Team Members:")
    st.write("**Lebogang** - Team Lead")
    st.write("**Phillip** - Project and Github Manager")
    st.write("**Sanele** - Kaggle Manager")
    st.write("**Tracy** - Trello Manager")
    st.write("**Matlou and Mzwandile** - Canvas Managers")

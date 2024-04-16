from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from surprise import Dataset, Reader, accuracy
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split
from surprise import SVD
import numpy as np
import pandas as pd
# Preprocessing
# Load the data
data_raw = pd.read_csv('Fix_Dataset/asuransi_clean.csv')
data_raw.head()
# membuat user_id dari username dengan encoding labelencoder supaya unik dan hanya dimiliki oleh satu user saja
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data_raw['user_id'] = le.fit_transform(data_raw['username'])

data_raw.head()
data_raw['user_id'].value_counts().head()
# mengubah rate menjadi 1-5
data_raw['rate'] = data_raw['rate'].map({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
data_raw['rate'].value_counts()
df = data_raw.copy()
# encoding data pada kolom Produk
label = { 'AIA' : 0, 'Allianz' : 1, 'Prudential' : 2, 'BNI Life' : 3, 'Manulife' : 4, 'Cigna' : 5 }

df['label'] = df['Produk'].replace(label)
df.head()
# Modelling
# langkah pertama sebelum membuat model Collaborative Filtering adalah membuat data menjadi format yang sesuai dengan library surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'label', 'rate']], reader)

# split data menjadi data train dan data test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# membuat model Collaborative Filtering dengan menggunakan SVD
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
# training model
model.fit(trainset)
# membuat prediksi
predictions = model.test(testset)
# menghitung error model dengan menggunakan MAE
mae = accuracy.mae(predictions)
print('MAE: ', mae)
# membuat fungsi untuk merekomendasikan produk asuransi
def recommend_product(user_id, model, n=5):
    # membuat list produk yang sudah direview oleh user
    produk_reviewed = df[df['user_id'] == user_id]['label'].unique()
    
    # membuat list produk yang belum direview oleh user
    produk_not_reviewed = [produk for produk in df['label'].unique() if produk not in produk_reviewed]
    
    # membuat prediksi rating untuk produk yang belum direview oleh user
    predictions = [model.predict(user_id, produk) for produk in produk_not_reviewed]
    
    # mengurutkan produk berdasarkan prediksi rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # mengambil n produk teratas
    top_n = predictions[:n]
    
    # mendapatkan id produk
    top_n_produk = [pred.iid for pred in top_n]
    
    # mendapatkan nama produk
    top_n_produk_name = [df[df['label'] == produk]['Produk'].values[0] for produk in top_n_produk]
    
    return top_n_produk_name
# membuat rekomendasi produk untuk user dengan id 
user_id = 15
recommend_product(user_id, model)
# menggunakan text review untuk merekomendasikan produk asuransi
# membuat model TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data_raw['full_text'])
# membuat fungsi untuk merekomendasikan produk asuransi berdasarkan text review membandingkan word2vec dan tfidf
# untuk text review yang diberikan oleh user menggunakan tfidf
def recommend_product_tfidf(text_review, model, n=5):
    # membuat vektor tfidf dari text review yang diberikan oleh user
    tfidf_vector = tfidf.transform([text_review])
    
    # menghitung cosine similarity antara vektor tfidf user dengan vektor tfidf produk
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_vector)
    
    # mengurutkan produk berdasarkan cosine similarity
    produk_index = cosine_sim.flatten().argsort()[::-1]
    
    # mengambil n produk teratas
    top_n = produk_index[:n]
    
    # mendapatkan nama produk
    top_n_produk_name = data_raw.iloc[top_n]['Produk'].unique()
    
    return top_n_produk_name

# membuat rekomendasi produk untuk user dengan text review
text_review = 'rekomendasi asuransi yang murah dan bagus untuk keluarga'
recommend_product_tfidf(text_review, model)
# deploy model dengan menggunakan pickle
import pickle
# save model
with open('Model/model_cf.pkl', 'wb') as file:
    pickle.dump(model, file)
# save tfidf
with open('Model/tfidf.pkl', 'wb') as file:
    pickle.dump(tfidf, file)
# test load model
with open('Model/model_cf.pkl', 'rb') as file:
    model_pkl_cf = pickle.load(file)

# test load tfidf
with open('Model/tfidf.pkl', 'rb') as file:
    tfidf_pkl = pickle.load(file)
# membuat rekomendasi produk untuk user dengan id
user_id = 15
recommend_product(user_id, model_pkl_cf)
# membuat rekomendasi produk untuk user dengan text review
text_review = 'rekomendasi asuransi yang murah dan bagus untuk keluarga'
recommend_product_tfidf(text_review, model_pkl_cf)
# membuat sistem rekomendasi produk asuransi menggunakan streamlit
import streamlit as st
import pickle

# load model
with open('Model/model_cf.pkl', 'rb') as file:
    model_pkl_cf = pickle.load(file)

# load tfidf
with open('Model/tfidf.pkl', 'rb') as file:
    tfidf_pkl = pickle.load(file)

# membuat judul
st.title('Rekomendasi Produk Asuransi')

# membuat input text review
text_review = st.text_input('Masukkan Text Review')

# membuat tombol untuk merekomendasikan produk
if st.button('Rekomendasi Produk'):
    # merekomendasikan produk asuransi berdasarkan text review
    recommended_product = recommend_product_tfidf(text_review, model_pkl_cf)

    # menampilkan rekomendasi produk
    st.write('Rekomendasi Produk:')
    for i, produk in enumerate(recommended_product):
        st.write(f'{i+1}. {produk}')
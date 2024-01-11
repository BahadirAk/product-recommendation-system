import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import recall_score

columns=['gameId', 'userId', 'ratings','timestamp']
games_df = pd.read_csv('ratings/game_ratings.csv', names=columns)

games_df.drop('timestamp', axis=1, inplace=True)

# Filter users with less than 15 ratings
counts = games_df.userId.value_counts()
games_df_final = games_df[games_df.userId.isin(counts[counts >= 15].index)]
games_df_final = games_df_final.drop_duplicates(subset=['userId'])

# Constructing the pivot table
final_ratings_matrix = games_df_final.pivot(index='userId', columns='gameId', values='ratings').fillna(0)

# Neural Collaborative Filtering (NCF)
def create_ncf_model(num_users, num_items, embedding_size=10, hidden_layer_size=20):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, input_length=1)(item_input)

    user_flatten = Flatten()(user_embedding)
    item_flatten = Flatten()(item_embedding)

    concat = Concatenate()([user_flatten, item_flatten])
    hidden_layer = Dense(hidden_layer_size, activation='relu')(concat)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=[user_input, item_input], outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create user and item indices for NCF
user_indices = final_ratings_matrix.index.astype('category').codes
item_indices = final_ratings_matrix.columns.astype('category').codes

pivot_df = final_ratings_matrix
pivot_df_sparse = csr_matrix(pivot_df)

# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df_sparse, k=10)
sigma = np.diag(sigma)

user_indices, item_indices = np.meshgrid(user_indices, item_indices)
user_indices = user_indices.flatten()
item_indices = item_indices.flatten()
labels = np.ones_like(user_indices, dtype=np.float32)

# Creating and training the NCF model
ncf_model = create_ncf_model(num_users=len(pivot_df), num_items=len(pivot_df.columns))
ncf_model.fit([user_indices, item_indices], labels, epochs=10, batch_size=64, validation_split=0.1)

ncf_predictions = ncf_model.predict([user_indices, item_indices]).flatten()

alpha = 0.5

# Blending SVD and NCF predictions
svd_predictions = np.dot(np.dot(U, sigma), Vt)
svd_predictions_df = pd.DataFrame(svd_predictions, index=pivot_df.index, columns=pivot_df.columns)

# NCF predictions
ncf_predictions = ncf_model.predict([user_indices, item_indices]).flatten()
ncf_predictions_reshaped = ncf_predictions.reshape(pivot_df.shape)  # Reshape to match the shape of pivot_df
ncf_predictions_df = pd.DataFrame(ncf_predictions_reshaped, index=pivot_df.index, columns=pivot_df.columns)

# Combine SVD and NCF predictions
combined_predictions = alpha * svd_predictions_df.values + (1 - alpha) * ncf_predictions_df.values

ground_truth = pivot_df.values.flatten()
svd_preds_flat = svd_predictions_df.values.flatten()
ncf_preds_flat = ncf_predictions_df.values.flatten()
combined_preds_flat = combined_predictions.flatten()

rmse_svd = sqrt(mean_squared_error(ground_truth, svd_preds_flat))
rmse_ncf = sqrt(mean_squared_error(ground_truth, ncf_preds_flat))
rmse_combined = sqrt(mean_squared_error(ground_truth, combined_preds_flat))

print('RMSE for SVD: {:.4f}'.format(rmse_svd))
print('RMSE for NCF: {:.4f}'.format(rmse_ncf))
print('RMSE for Combined SVD and NCF: {:.4f}'.format(rmse_combined))

aligned_ncf_predictions = ncf_predictions.reshape(pivot_df.shape)
final_predictions = alpha * svd_predictions_df.values + (1 - alpha) * aligned_ncf_predictions

final_preds_df = pd.DataFrame(final_predictions, index=pivot_df.index, columns=pivot_df.columns)

# User-User Collaborative Filtering
user_similarity = cosine_similarity(final_ratings_matrix)

def predict_user_user_cf(user_id, user_similarity, final_ratings_matrix):

    user_index = final_ratings_matrix.index.get_loc(user_id)
    
    similar_users = user_similarity[user_index]
    
    rated_items = final_ratings_matrix.loc[user_id].values.astype(bool)
    
    similar_users = similar_users[:len(rated_items)]
    
    rated_users = similar_users[rated_items]
    
    predicted_ratings = final_ratings_matrix.iloc[rated_users].mean(axis=0)
    
    return predicted_ratings

# Calculate RMSE for User-User Collaborative Filtering
user_user_preds = np.zeros_like(final_ratings_matrix.values, dtype=np.float32)

for user_index in range(len(final_ratings_matrix.index)):
    user_id = final_ratings_matrix.index[user_index]
    user_user_preds[user_index, :] = predict_user_user_cf(user_id, user_similarity, final_ratings_matrix).values

rmse_user_user = sqrt(mean_squared_error(ground_truth, user_user_preds.flatten()))
print('RMSE for User-User Collaborative Filtering: {:.4f}'.format(rmse_user_user))

beta = 0.3

user_user_preds_df = pd.DataFrame(user_user_preds, index=pivot_df.index, columns=pivot_df.columns)

final_combined_predictions = alpha * svd_predictions_df.values + (1 - alpha) * aligned_ncf_predictions + beta * user_user_preds_df.values

final_combined_preds_flat = final_combined_predictions.flatten()

# Calculate RMSE for the combined model
rmse_final_combined = sqrt(mean_squared_error(ground_truth, final_combined_preds_flat))
print('RMSE for Combined SVD, NCF, and User-User Collaborative Filtering: {:.4f}'.format(rmse_final_combined))

threshold = 4.0

ncf_binary_preds = (ncf_preds_flat >= threshold).astype(int)
user_user_binary_preds = (user_user_preds.flatten() >= threshold).astype(int)

ground_truth_binary = (ground_truth >= threshold).astype(int)

# Calculate accuracy
accuracy_ncf = np.sum(ncf_binary_preds == ground_truth_binary) / len(ground_truth)
accuracy_user_user = np.sum(user_user_binary_preds == ground_truth_binary) / len(ground_truth)

print('Accuracy for NCF: {:.4f}'.format(accuracy_ncf))
print('Accuracy for User-User Collaborative Filtering: {:.4f}'.format(accuracy_user_user))

# Calculate recall
recall_ncf = recall_score(ground_truth_binary, ncf_binary_preds)
recall_user_user = recall_score(ground_truth_binary, user_user_binary_preds)

print('Recall for NCF: {:.4f}'.format(recall_ncf))
print('Recall for User-User Collaborative Filtering: {:.4f}'.format(recall_user_user))

def recommend_items_combined(user_id, final_preds_df, num_recommendations, alpha=0.5):

    user_predictions = final_preds_df.loc[user_id]
    
    # User-User Collaborative Filtering predictions
    cf_predictions = predict_user_user_cf(user_id, user_similarity, final_ratings_matrix)
    
    # Combine predictions
    combined_predictions = alpha * user_predictions + (1 - alpha) * pd.Series(cf_predictions)
    
    # Get top recommendations
    recommended_items = combined_predictions.sort_values(ascending=False).head(num_recommendations).index
    
    return recommended_items

user_id = 'BAHAD'
num_recommendations = 10

recommended_items = recommend_items_combined(user_id, final_preds_df, num_recommendations)
print('\nBelow are the recommended items for user (user_id = {}):\n'.format(user_id))
print(recommended_items)
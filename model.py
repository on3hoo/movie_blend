import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, MinMaxScaler

# =====================================================
# 1. LOAD DATA
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "imdbmovies.csv")
df = pd.read_csv(csv_path)

df = df[["Title", "Genre", "Description", "Director", "Actors", "Rating"]]
df.fillna("", inplace=True)

# =====================================================
# 2. CONTENT-BASED FEATURE ENGINEERING
# =====================================================
df["Content"] = (
    df["Genre"] + " " +
    df["Description"] + " " +
    df["Director"] + " " +
    df["Actors"]
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["Content"])

# =====================================================
# 3. THEME (GENRE ONLY)
# =====================================================
theme_vectorizer = TfidfVectorizer(stop_words="english")
theme_matrix = theme_vectorizer.fit_transform(df["Genre"])

# =====================================================
# 4. USER PROFILE CREATION (FIXED)
# =====================================================
def build_user_profile(user_movies):
    indices = df[df["Title"].isin(user_movies)].index
    if len(indices) == 0:
        raise ValueError("None of the user movies found in dataset.")

    # Convert to ndarray explicitly (IMPORTANT FIX)
    profile = np.asarray(tfidf_matrix[indices].mean(axis=0))

    profile = normalize(profile)

    return profile


def calculate_user_similarity(user1_movies, user2_movies):
    profile1 = build_user_profile(user1_movies)
    profile2 = build_user_profile(user2_movies)

    cosine_sim = cosine_similarity(profile1, profile2)[0][0]

    overlap = len(set(user1_movies) & set(user2_movies))
    overlap_boost = 0.10 * overlap

    adjusted_similarity = min(cosine_sim + overlap_boost, 1.0)

    return round(adjusted_similarity * 100, 2)


# =====================================================
# 5. THEME PROFILE (FIXED)
# =====================================================
def build_theme_profile(user_movies):
    indices = df[df["Title"].isin(user_movies)].index
    if len(indices) == 0:
        raise ValueError("None of the user movies found in dataset.")

    profile = np.asarray(theme_matrix[indices].mean(axis=0))

    profile = normalize(profile)

    return profile


def calculate_theme_similarity(user1_movies, user2_movies):
    profile1 = build_theme_profile(user1_movies)
    profile2 = build_theme_profile(user2_movies)

    cosine_sim = cosine_similarity(profile1, profile2)[0][0]

    return round(cosine_sim * 100, 2)


# =====================================================
# 6. SVM TRAINING
# =====================================================
def train_svm(user_movies):
    X = tfidf_matrix
    y = df["Title"].isin(user_movies).astype(int)

    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)
    return model


# =====================================================
# 7. HYBRID RECOMMENDATION (NORMALIZED)
# =====================================================
def recommend_common_movies(user1_movies, user2_movies, top_n=5):
    svm_user1 = train_svm(user1_movies)
    svm_user2 = train_svm(user2_movies)

    prob_user1 = svm_user1.predict_proba(tfidf_matrix)[:, 1]
    prob_user2 = svm_user2.predict_proba(tfidf_matrix)[:, 1]

    combined_score = (prob_user1 + prob_user2) / 2

    scaler = MinMaxScaler()
    combined_score = scaler.fit_transform(
        combined_score.reshape(-1, 1)
    ).flatten()

    watched_movies = set(user1_movies + user2_movies)

    recommendations = df.copy()
    recommendations["score"] = combined_score
    recommendations = recommendations[
        ~recommendations["Title"].isin(watched_movies)
    ]

    return recommendations.sort_values(
        by="score", ascending=False
    ).head(top_n)


# =====================================================
# 8. MAIN FUNCTION
# =====================================================
def get_recommendations(user1_movies, user2_movies, top_n=5):
    backend_similarity = calculate_user_similarity(
        user1_movies, user2_movies
    )

    theme_similarity = calculate_theme_similarity(
        user1_movies, user2_movies
    )

    recommended_movies = recommend_common_movies(
        user1_movies, user2_movies, top_n=top_n
    )

    return {
        "backend_similarity": backend_similarity,
        "theme_similarity": theme_similarity,
        "top_movies": recommended_movies[
            ["Title", "Genre", "Rating"]
        ].to_dict(orient="records")
    }


# =====================================================
# 9. TEST RUN
# =====================================================
if __name__ == "__main__":
    user1_movies = ["Home", "Moana", "Ice Age: Collision Course"]
    user2_movies = ["Ballerina", "Spider-Man 3", "Pitch Perfect"]

    print(get_recommendations(user1_movies, user2_movies))

'''# =====================================================
# 9. CALCULATE SIMILARITIES
# =====================================================
backend_similarity = calculate_user_similarity(user1_movies, user2_movies)
theme_similarity = calculate_theme_similarity(user1_movies, user2_movies)

# =====================================================
# 10. GET RECOMMENDATIONS
# =====================================================
recommended_movies = recommend_common_movies(user1_movies, user2_movies, top_n=5)

# =====================================================
# 11. OUTPUT RESULTS
# =====================================================
print("=" * 60)
print(f" Theme Taste Similarity (Genre Based): {theme_similarity:.2f}%")
#print(f" Internal Model Similarity (Used for AI predictions): {backend_similarity:.2f}%")
print("=" * 60)

print("\n Recommended Movies Both Users May Enjoy:\n")

for idx, row in recommended_movies.iterrows():
    print(f" {row['Title']} | Genre: {row['Genre']} | Rating: {row['Rating']}")

best_movie = recommended_movies.iloc[0]

print("\n BEST MOVIE RECOMMENDATION:")
print(f"{best_movie['Title']} — {best_movie['Genre']} — Rating: {best_movie['Rating']}")

# =====================================================
# 12. ANIMATED SPEEDOMETER VISUAL
# =====================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle,Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np

def animated_speedometer(score):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Color zones
    colors = ['#ff4d4d', '#ffa64d', '#ffff66', '#66ff66']
    ranges = [0, 25, 50, 75, 100]

    # Draw colored arcs
    for i in range(len(ranges)-1):
        wedge = Wedge((0, 0), 1,
                      180 - (ranges[i+1]*1.8),
                      180 - (ranges[i]*1.8),
                      facecolor=colors[i])
        ax.add_patch(wedge)

    # Draw center circle
    ax.add_patch(Circle((0, 0), 0.05, color='black'))

    # Title
    plt.title(" Movie Taste Match Meter", fontsize=16, weight='bold')
    ax.text(0, -0.28, f"Your overall taste similarity is {score:.2f}%", ha='center', fontsize=12, weight='bold')

    # Needle line
    needle, = ax.plot([], [], lw=3, color='black')

    # Score text
    score_text = ax.text(0, -0.2, "",
                         horizontalalignment='center',
                         fontsize=14, weight='bold')

    # Animation function
    def update(frame):
        angle = 180 - (frame * 1.8)
        x = 0.8 * np.cos(np.deg2rad(angle))
        y = 0.8 * np.sin(np.deg2rad(angle))
        needle.set_data([0, x], [0, y])
        score_text.set_text(f"{frame:.1f}% Match")
        return needle, score_text

    frames = np.linspace(0, score, 100)

    ani = FuncAnimation(fig, update, frames=frames,
                        interval=20, blit=True, repeat=False)

    plt.show()


# Run animated speedometer
animated_speedometer(theme_similarity)


# -------------------------------
# RECOMMENDATION VISUAL WINDOW
# -------------------------------

def show_recommendations_window(recommended_movies):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    plt.title(" Movies You Both May Love!", fontsize=18, weight='bold')

    y_positions = np.linspace(0.8, 0.2, len(recommended_movies))

    for y, (_, row) in zip(y_positions, recommended_movies.iterrows()):
        # Draw card rectangle
        rect = Rectangle((0.1, y-0.05), 0.8, 0.1,
                         edgecolor='black',
                         facecolor='#f0f8ff',
                         linewidth=2)
        ax.add_patch(rect)

        # Movie text
        text = f"{row['Title']}  |  {row['Genre']}  |  {row['Rating']}"
        ax.text(0.12, y, text, fontsize=12, verticalalignment='center')

    plt.show()
show_recommendations_window(recommended_movies)'''





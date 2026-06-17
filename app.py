from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from model import get_recommendations

app = Flask(__name__)
CORS(app)  # allow frontend requests

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    user1_movies = data.get("user1", [])
    user2_movies = data.get("user2", [])

    if not user1_movies or not user2_movies:
        return jsonify({"error": "Both users must provide movies"}), 400

    try:
        results = get_recommendations(user1_movies, user2_movies, top_n=5)

        return jsonify({
            "backend_similarity": round(results["backend_similarity"], 2),
            "theme_similarity": round(results["theme_similarity"], 2),
            "recommendations": results["top_movies"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware

# ---------- Step 1: Create FastAPI App ----------
app = FastAPI(title="Enhanced Fake Account Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Step 2: Define Input Schema ----------
class AccountData(BaseModel):
    username: str
    followers: int
    following: int
    posts: int
    bio_length: int


# ---------- Step 3: Train Data ----------
# Features: [followers, following, posts, bio_length]

fake_accounts = [
    [2, 1000, 1, 0],
    [5, 800, 2, 1],
    [10, 900, 3, 2],
    [15, 700, 5, 5],
    [20, 600, 4, 3],
    [25, 500, 6, 4],
    [10, 400, 3, 2],
    [5, 450, 2, 0],
    [12, 600, 4, 3],
    [8, 550, 1, 1],
    [18, 700, 5, 4],
    [30, 800, 10, 5],
    [22, 650, 8, 4],
    [15, 500, 6, 3],
    [7, 300, 3, 1],
    [9, 400, 2, 0],
    [10, 700, 1, 0],
    [20, 1000, 3, 1],
    [25, 900, 6, 2],
    [5, 1200, 2, 0],
    [12, 950, 3, 1],
    [8, 1000, 1, 0],
    [15, 800, 4, 2],
    [2, 600, 1, 0],
    [3, 700, 2, 0],
    [10, 750, 3, 1],
    [18, 600, 4, 2],
    [25, 500, 5, 3],
    [7, 1000, 2, 1],
    [5, 1100, 1, 0],
    [20, 900, 3, 1],
    [10, 850, 2, 1],
    [6, 700, 1, 0],
    [9, 1000, 2, 0],
    [4, 500, 1, 0],
    [10, 1200, 2, 1],
    [15, 900, 3, 1],
    [8, 800, 1, 0],
    [25, 700, 5, 3],
    [18, 600, 3, 2],
    [12, 500, 2, 1],
    [15, 1000, 3, 1],
    [5, 400, 1, 0],
    [20, 900, 4, 2],
    [8, 750, 2, 0],
    [10, 800, 1, 0],
    [6, 950, 1, 0],
    [4, 700, 1, 0],
    [22, 1000, 4, 2],
    [12, 900, 3, 1],
    [9, 850, 2, 0],
    [2, 50, 0, 3],
    [2, 41, 0, 5],
    [7, 30, 0, 2],
    [10, 40, 0, 4],
    [7, 30, 0, 5],
    [10, 1000, 0, 10],
]

real_accounts = [
    [63, 170, 20, 40],     
    [100, 200, 50, 80],
    [150, 250, 60, 100],
    [200, 300, 120, 200],
    [300, 400, 150, 250],
    [400, 500, 200, 300],
    [500, 600, 300, 400],
    [75, 100, 15, 25],
    [80, 120, 20, 30],
    [90, 180, 25, 40],
    [120, 150, 30, 50],
    [130, 200, 40, 70],
    [140, 180, 35, 60],
    [160, 220, 45, 90],
    [180, 240, 55, 110],
    [220, 300, 70, 130],
    [250, 350, 90, 150],
    [270, 400, 120, 180],
    [320, 450, 150, 220],
    [350, 480, 180, 250],
    [380, 500, 200, 280],
    [420, 550, 250, 300],
    [450, 600, 300, 350],
    [480, 650, 320, 370],
    [520, 700, 350, 400],
    [560, 750, 400, 450],
    [600, 800, 450, 500],
    [650, 850, 500, 520],
    [700, 900, 550, 540],
    [800, 1000, 600, 600],
    [85, 170, 18, 35],
    [95, 190, 20, 40],
    [110, 200, 22, 45],
    [125, 210, 25, 55],
    [140, 250, 30, 65],
    [160, 270, 35, 80],
    [180, 300, 40, 90],
    [200, 320, 45, 100],
    [220, 350, 50, 120],
    [240, 370, 60, 140],
    [260, 390, 70, 160],
    [280, 410, 80, 180],
    [300, 430, 90, 200],
    [320, 450, 100, 220],
    [340, 470, 110, 240],
    [360, 490, 120, 260],
    [380, 510, 130, 280],
    [400, 530, 140, 300],
    [420, 550, 150, 320],
    [440, 570, 160, 340],
    [63, 200, 0, 5],
    [338, 312, 3, 0],
    [150, 500, 20, 10],
]

# Combine datasets
X_train = np.array(fake_accounts + real_accounts)
y_train = np.array([1] * len(fake_accounts) + [0] * len(real_accounts))

# ---------- Step 4: Train Model ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ---------- Step 5: Prediction Endpoint ----------
@app.post("/predict")
def predict(data: AccountData):
    try:
        features = np.array([[data.followers, data.following, data.posts, data.bio_length]])
        features_scaled = scaler.transform(features)
        prob_fake = model.predict_proba(features_scaled)[0][1]
        prediction = "Fake" if prob_fake >= 0.5 else "Real"

        return {
            "username": data.username,
            "prediction": prediction,
            "probability_fake": round(float(prob_fake), 2)
        }

    except Exception as e:
        return {"error": str(e)}

# ---------- Step 6: Run the Server ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

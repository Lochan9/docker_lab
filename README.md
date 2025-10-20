
- **model_training.py** → Trains and saves the model (`wine_model.keras`) and scaler (`scaler.pkl`)  
- **main.py** → Flask web server that loads model and handles predictions  
- **predict.html** → Frontend UI for entering 11 features and viewing results  
- **dockerfile** → Multi-stage build for training and serving  
- **requirements.txt** → Python dependencies list  

---

## 🧪 How It Works

1. **Stage 1 (Model Training)**  
   - Trains a deep learning model inside Docker.  
   - Saves model as `wine_model.keras` and scaler as `scaler.pkl`.  

2. **Stage 2 (Model Serving)**  
   - Loads trained artifacts into Flask app.  
   - Runs a lightweight prediction API and web interface.

3. **User Interaction**  
   - User enters wine features in browser.  
   - Model predicts wine quality (score between 0–10).  

---

## 🧰 Technologies Used

| Layer | Tools |
|-------|--------|
| **Language** | Python 3.9 |
| **Framework** | Flask |
| **ML Library** | TensorFlow (CPU) |
| **Scaler** | scikit-learn |
| **Frontend** | HTML5, CSS3 |
| **Containerization** | Docker |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Lochan9/docker_lab.git
cd docker_lab/scr
docker build -t wine_app .
docker run -p 4000:4000 wine_app
Example Input (Wine Features)
Feature	Example Value
Fixed Acidity	7.4
Volatile Acidity	0.7
Citric Acid	0.0
Residual Sugar	1.9
Chlorides	0.076
Free Sulfur Dioxide	11
Total Sulfur Dioxide	34
Density	0.9978
pH	3.51
Sulphates	0.56
Alcohol	9.4

🧮 Predicted Quality: Around 4.5 – 5.0

docker build -t wine_app .

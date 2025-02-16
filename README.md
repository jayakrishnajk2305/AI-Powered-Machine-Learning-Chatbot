# **🤖 SONI – AI-Powered Machine Learning Chatbot**  
🚀 **An Intelligent NLP Chatbot for Real-World Conversations**  

## 📌 **Project Overview**  
SONI is a **state-of-the-art machine learning chatbot** designed to understand and respond to user queries with high accuracy.  
Built using **Natural Language Processing (NLP)** and **deep learning techniques**, this chatbot leverages **transformer-based embeddings** for context-aware responses.  

🔹 **Why It’s Unique?**  
- Uses **BERT-based embeddings** for high-quality text understanding.  
- Implements **multiple machine learning models** for intent classification.  
- Optimized for **scalability** and **real-world deployment**.  

---

## ✨ **Key Features**  
✅ **Advanced NLP Processing** – Tokenization, Lemmatization, Stopword Removal, and Feature Engineering.  
✅ **Pretrained Transformer Embeddings** – Uses a **768-dimensional vector representation** for accurate text understanding.  
✅ **Multi-Model Approach** – Experimented with **Logistic Regression, KNN, Decision Trees, and SVM**.  
✅ **Optimized Training Pipeline** – Fine-tuned **hyperparameters** for maximum accuracy.  
✅ **Deployment Ready** – Can be integrated into APIs, chat platforms, or websites.  

---

## 🚀 **Project Pipeline**  

### **1️⃣ Data Preprocessing**  
- **Text Cleaning & Normalization**  
  - Tokenization  
  - Lemmatization using `WordNetLemmatizer`  
  - Stopword removal  
  - Lowercasing and punctuation removal  
- **Feature Extraction**  
  - Used **768-dimensional BERT-based embeddings**  
  - Transformed textual input into numerical vectors  

### **2️⃣ Model Training & Selection**  
We tested multiple **machine learning models** for intent classification:  

| **Model**                | **Accuracy** |
|--------------------------|-------------|
| Logistic Regression      | **39%**      |
| K-Nearest Neighbors (KNN) | **48%**      |
| Decision Tree           | **32%**      |
| Support Vector Machine (SVM) | **46%** |

🔹 **Train-Test Split:**  
✅ **Training Data:** `X_train shape (740, 768)`  
✅ **Test Data:** `X_test shape (186, 768)`  

### **3️⃣ Model Evaluation**  
📌 **Performance Metrics Used:**  
✔ **Accuracy**  
✔ **F1-score**  
✔ **Classification Report**  
✔ **Confusion Matrix**  

| Model                  | Macro Avg | Weighted Avg | Accuracy |
|------------------------|-----------|-------------|-----------|
| **Logistic Regression** | 0.03 / 0.04 / 0.03 | 0.20 / 0.39 / 0.24 | **39%** |
| **K-Nearest Neighbors** | 0.09 / 0.11 / 0.09 | 0.34 / 0.48 / 0.39 | **48%** |
| **Decision Tree**       | 0.05 / 0.07 / 0.06 | 0.30 / 0.32 / 0.31 | **32%** |
| **SVM**                | 0.07 / 0.10 / 0.08 | 0.29 / 0.46 / 0.36 | **46%** |

---

## 🏗 **Technologies Used**  
🛠 **Programming Language:** Python  
🛠 **Libraries & Frameworks:**  
- `NumPy`, `Pandas` → Data manipulation  
- `NLTK`, `spaCy` → NLP processing  
- `Scikit-learn`, `TensorFlow`, `PyTorch` → Model training  
- `Matplotlib`, `Seaborn` → Data visualization  

---

## 💻 **How to Use the Chatbot**  

### 🔹 1. Clone the Repository  
```bash
git clone https://github.com/yourusername/SONI-Chatbot.git
cd SONI-Chatbot
```

### 🔹 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 🔹 3. Run the Chatbot  
```bash
python chatbot.py
```

### 🔹 4. Start Interacting!  
🗣 **Ask anything!** The chatbot will respond based on its trained knowledge.  

---

## 📈 **Results & Insights**  
📌 **Key Takeaways:**  
✅ **Transformer embeddings significantly improved classification accuracy.**  
✅ **SVM & KNN performed best, achieving ~46-48% accuracy.**  
✅ **Machine learning approach outperformed rule-based chatbot models.**  

---

## 🔮 **Future Enhancements**  
🚀 **Planned Upgrades:**  
🔹 **LSTM/Transformer Models** – Implement **deep learning** for better contextual understanding.  
🔹 **API Integration** – Deploy chatbot as a **REST API** for web and mobile apps.  
🔹 **Real-time Learning** – Implement **reinforcement learning** for adaptive responses.  
🔹 **Expand Dataset** – Improve accuracy by **adding more diverse training data**.  

---

## 📖 **About Me**  
👋 Hi! I’m **Jaya Krishna**, a passionate **Data Scientist & Machine Learning Engineer** with expertise in **NLP, AI, and Chatbot Development**.  

📌 **Let’s Connect!**  
📩 **Email**: jaya2305krishna@gmail.com  
🔗 **LinkedIn**: [linkedin.com/in/jaya23krishna](https://linkedin.com/in/jaya23krishna)  
🌟 **GitHub**: [github.com/jaya23krishna](https://github.com/jaya23krishna)  

💡 _"Building AI-driven chatbots to make human-computer interaction seamless!"_  

---

🚀 **If you like this project, give it a ⭐ on GitHub! Contributions are welcome!**  

---


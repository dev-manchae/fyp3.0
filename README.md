# 🎮 Steam Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Model-RoBERTa-yellow?logo=huggingface&logoColor=white)

A real-time NLP dashboard that analyzes player sentiment from Steam game reviews. This project utilizes a fine-tuned **RoBERTa** model to classify reviews as **Satisfied**, **Neutral**, or **Dissatisfied**, providing game developers and analysts with deep insights into player feedback.

🔗 **[Live Demo](https://sentanalygames.streamlit.app/)** *(Replace this with your actual Streamlit App URL)*

---

## ✨ Key Features

### 1. 📊 Interactive Analytics Dashboard
* **Sentiment Distribution:** Visualize the ratio of positive vs. negative reviews using interactive Pie Charts.
* **Time Series Analysis:** Track how sentiment changes over time (Daily/Monthly) with smart zooming capabilities.
* **Game Comparison:** Compare satisfaction rates across different games in the dataset.

### 2. ☁️ Smart Word Clouds
* **Advanced Cleaning:** Filters out domain-specific noise (e.g., "game", "play", "steam") and common stop words to reveal meaningful topics.
* **Sentiment Filtering:** Generate separate clouds for *Satisfied* vs. *Dissatisfied* users to see exactly what they love or hate.

### 3. 🤖 Live AI Sentiment Lab
* **Real-time Inference:** Type any fake review and get an instant classification from the AI.
* **Confidence Scoring:** See the model's certainty percentage for each prediction.
* **Hybrid Architecture:** The app runs on Streamlit Cloud but fetches the heavy model weights dynamically from **Hugging Face**, ensuring fast performance without exceeding storage limits.

---

## 🛠️ Architecture

* **Frontend:** [Streamlit](https://streamlit.io/) (Python)
* **Data Visualization:** Plotly Express & Matplotlib
* **Machine Learning:** Transformers (Hugging Face), PyTorch
* **Model:** Custom Fine-Tuned RoBERTa (Hosted on Hugging Face Hub)
* **Data:** Pre-processed CSV dataset (`STEAM_REVIEWS_3_CLASS_ROBERTA.csv`)

---

## 🚀 How to Run Locally

If you want to run this dashboard on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/dev-manchae/fyp3.0.git](https://github.com/dev-manchae/fyp3.0.git)
cd fyp3.0

**2. Install dependencies Make sure you have Python installed. Then run:**

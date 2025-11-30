import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import re
import ftfy
from unidecode import unidecode

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Steam Sentiment Dashboard", layout="wide", page_icon="🎮")

# --- PATH CONFIGURATION ---
DATA_FILE = "STEAM_REVIEWS_3_CLASS_ROBERTA.csv"
BENCHMARK_CSV = "v4_model_metrics.csv"
# Removed BENCHMARK_IMG because we are building it with code now!
MODEL_PATH = "manchae86/steam-sentiment-model" 

# --- HELPER: TEXT CLEANING ---
def clean_text_for_model(text):
    if not isinstance(text, str): return ""
    text = ftfy.fix_text(text)
    text = unidecode(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- LOAD DATA ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    
    label_map = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
    if 'sentiment_label' in df.columns:
        df['Sentiment'] = df['sentiment_label'].map(label_map)
    
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    if 'clean_text' in df.columns:
        df['clean_text'] = df['clean_text'].astype(str).fillna("")
        
    return df

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

df = load_data()

if df is None:
    st.error(f"❌ **File Not Found:** `{DATA_FILE}`")
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🎮 Menu")

menu_choice = st.sidebar.radio(
    "Go to:",
    ["📊 Analytics Dashboard", "☁️ Word Clouds", "🤖 Live AI Tester", "🏆 Model Benchmarks"]
)

st.sidebar.divider()

# Global Filters
st.sidebar.header("Filter Options")
all_games = ["All Games"] + list(df['game_name'].unique())
selected_game = st.sidebar.selectbox("Select a Game:", all_games)

if selected_game != "All Games":
    filtered_df = df[df['game_name'] == selected_game]
else:
    filtered_df = df

# Global Metrics
st.sidebar.divider()
total = len(filtered_df)
st.sidebar.metric("Total Reviews", f"{total:,}")
if total > 0:
    sat_rate = len(filtered_df[filtered_df['Sentiment']=='Satisfied']) / total * 100
    st.sidebar.metric("Satisfaction", f"{sat_rate:.1f}%")

# ==========================================
# PAGE 1: ANALYTICS DASHBOARD
# ==========================================
if menu_choice == "📊 Analytics Dashboard":
    st.title(f"📊 Analytics: {selected_game}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        if len(filtered_df) > 0:
            counts = filtered_df['Sentiment'].value_counts().reset_index()
            counts.columns = ['Sentiment', 'Count']
            fig_pie = px.pie(counts, values='Count', names='Sentiment', 
                             color='Sentiment', 
                             color_discrete_map={'Satisfied':'#66c2a5', 'Neutral':'#fc8d62', 'Dissatisfied':'#e74c3c'},
                             hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available.")

    with col2:
        st.subheader("Sentiment Trend")
        if 'date' in filtered_df.columns and not filtered_df['date'].isna().all():
            min_date = filtered_df['date'].min()
            max_date = filtered_df['date'].max()
            days_diff = (max_date - min_date).days
            
            if days_diff < 60:
                time_period = 'D'
                x_label = "Date (Daily)"
            else:
                time_period = 'M'
                x_label = "Date (Monthly)"

            plot_df = filtered_df.copy()
            plot_df['time_group'] = plot_df['date'].dt.to_period(time_period).astype(str)
            
            timeline = plot_df.groupby(['time_group', 'Sentiment']).size().reset_index(name='Count')
            
            if len(timeline) < 2:
                st.warning("Not enough time data to plot a trend line.")
            else:
                timeline = timeline.sort_values('time_group')
                fig_line = px.line(timeline, x='time_group', y='Count', color='Sentiment',
                                   color_discrete_map={'Satisfied':'#66c2a5', 'Neutral':'#fc8d62', 'Dissatisfied':'#e74c3c'},
                                   markers=True)
                fig_line.update_layout(xaxis_title=x_label)
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("No timestamp data found.")

    if selected_game == "All Games":
        st.subheader("Comparison: Which Game is Winning?")
        comp_df = pd.crosstab(df['game_name'], df['Sentiment'], normalize='index').reset_index()
        comp_df = comp_df.sort_values(by="Satisfied", ascending=True)
        fig_bar = px.bar(comp_df, y='game_name', x=['Dissatisfied', 'Neutral', 'Satisfied'],
                         orientation='h',
                         color_discrete_map={'Satisfied':'#66c2a5', 'Neutral':'#fc8d62', 'Dissatisfied':'#e74c3c'},
                         title="100% Stacked Sentiment")
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 2: WORD CLOUDS
# ==========================================
elif menu_choice == "☁️ Word Clouds":
    st.title(f"☁️ Word Cloud Analysis: {selected_game}")
    st.write("See the most common words used in reviews.")
    
    sentiment_filter = st.radio("Select Sentiment to Analyze:", ["Satisfied", "Dissatisfied"], horizontal=True)
    
    subset = filtered_df[filtered_df['Sentiment'] == sentiment_filter]
    
    if subset.empty:
        st.warning(f"No {sentiment_filter} reviews found for {selected_game}.")
    else:
        with st.spinner("Cleaning text and generating Word Cloud..."):
            text_data = " ".join(subset['clean_text'].tolist()).lower()

            # Visual Cleaning
            text_data = re.sub(r'https?://\S+|www\.\S+', '', text_data)
            text_data = re.sub(r'<.*?>', '', text_data)
            text_data = re.sub(r'&[a-z]+;', '', text_data)
            text_data = re.sub(r'[^a-z\s]', '', text_data)
            text_data = re.sub(r'\b\w{1,2}\b', '', text_data)

            if len(text_data) < 50:
                st.warning("Not enough text data after cleaning to generate a word cloud.")
            else:
                final_stopwords = set(STOPWORDS)
                custom_words = [
                    'game', 'play', 'played', 'playing', 'player', 'hours', 
                    'review', 'steam', 'access', 'early', 'version',
                    'make', 'get', 'go', 'do', 'did', 'done', 'just', 'really', 
                    'very', 'much', 'even', 'also', 'thing', 'one', 'would', 
                    'could', 'should', 'good', 'bad', 'people', 'think', 'know',
                    'time', 'day', 'year', 'best', 'better', 'great'
                ]
                final_stopwords.update(custom_words)

                wc = WordCloud(
                    width=1000, height=500, background_color='white', 
                    stopwords=final_stopwords, colormap='viridis', collocations=False
                ).generate(text_data)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
                st.success(f"Generated from {len(subset)} reviews.")

# ==========================================
# PAGE 3: LIVE AI TESTER
# ==========================================
elif menu_choice == "🤖 Live AI Tester":
    st.title("🤖 Live AI Sentiment Lab")
    st.markdown("Test your custom model with new text.")
    
    user_input = st.text_area("Write a fake review here:", "The graphics are amazing but the servers keep crashing.")
    
    if st.button("Analyze Sentiment"):
        tokenizer, model = load_model()
        
        if model is None:
            st.error(f"❌ **Model Not Found:** `{MODEL_PATH}`")
        else:
            with st.spinner("Processing..."):
                cleaned_input = clean_text_for_model(user_input)
                inputs = tokenizer(cleaned_input, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_idx].item()
                
                labels = ["Dissatisfied 😡", "Neutral 😐", "Satisfied 😃"]
                colors = ["#e74c3c", "#fc8d62", "#66c2a5"]
                
                st.markdown(f"### Prediction: <span style='color:{colors[pred_idx]}'>{labels[pred_idx]}</span>", unsafe_allow_html=True)
                
                with st.expander("See how the AI sees your text"):
                    st.code(cleaned_input)

                st.progress(confidence)
                st.caption(f"Confidence Score: {confidence:.1%}")
                
                chart_data = pd.DataFrame({
                    "Sentiment": ["Dissatisfied", "Neutral", "Satisfied"],
                    "Probability": probs[0].tolist()
                })
                st.bar_chart(chart_data.set_index("Sentiment"))

# ==========================================
# PAGE 4: MODEL BENCHMARKS (DYNAMIC)
# ==========================================
elif menu_choice == "🏆 Model Benchmarks":
    st.title("🏆 Model Performance & Comparison")
    st.markdown("Comparing our fine-tuned **RoBERTa v4.0** against industry baselines.")
    
    if os.path.exists(BENCHMARK_CSV):
        # 1. Load Data
        metrics_df = pd.read_csv(BENCHMARK_CSV, index_col=0)
        
        # 2. Display Raw Stats
        st.subheader("📋 Statistical Performance")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#66c2a5'), use_container_width=True)

        st.divider()

        # 3. Create Dynamic Plotly Chart
        st.subheader("📊 Accuracy vs. F1-Score Comparison")
        
        # Prepare data for Plotly (Melt from Wide to Long format)
        # Result: Columns will be [Model, Metric, Score]
        plot_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
        plot_df = plot_df.melt(id_vars='Model', 
                               value_vars=['Accuracy', 'F1-Score'], 
                               var_name='Metric', 
                               value_name='Score')
        
        # Sort by Score so the best model is on the right
        plot_df = plot_df.sort_values(by='Score', ascending=True)

        # Generate Chart
        fig = px.bar(
            plot_df, 
            x='Model', 
            y='Score', 
            color='Metric', 
            barmode='group',
            text_auto='.3f', # Show values on bars automatically
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Comparison: VADER vs SVM vs RoBERTa"
        )
        
        # Customize Layout
        fig.update_layout(
            yaxis_title="Score (0-1)",
            yaxis_range=[0, 1.1],
            xaxis_title="Model Type",
            legend_title="Metric"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"Benchmark CSV ({BENCHMARK_CSV}) not found. Please upload it to the GitHub repository.")

    st.divider()

    # 4. Key Findings
    st.subheader("🧠 Key Findings")
    st.markdown("""
    * **RoBERTa (v4.0) is the Champion:** Achieving **~93.7% accuracy**, it significantly outperforms traditional methods.
    * **Why VADER Failed (74.7%):** As a rule-based lexicon, VADER struggled with **gaming slang** (e.g., "nerf", "grind", "broken") and sarcasm.
    * **Why RoBERTa Won:** Unlike SVM (which looks at word frequency), RoBERTa understands **context**. It knows that *"long hours"* is good for an RPG but bad for a loading screen.
    """)
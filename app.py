import streamlit as st
import pandas as pd
import altair as alt
import os
from openai import OpenAI

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("sample_dataset.csv", encoding="latin1")

df = load_data()

# Filter years 2000–2025
df = df[(df['funding_round_year'] >= 2000) & (df['funding_round_year'] <= 2025)]

# Title and description
st.title("Global Startup Investment Explorer - AI-powered")
st.markdown("""
**App Description:**  
This is an AI-powered app to explore a curated sample of global startup investments. It provides a mockup for querying investment data across ecosystems.
""")

# KPIs
st.subheader("Key Performance Indicators")
st.write("**Rows:**", df.shape[0])
st.write("**Columns:**", df.shape[1])
if 'funding_round_amount_usd_per_invsestor' in df.columns:
    st.metric("Total Investment (USD)", f"${df['funding_round_amount_usd_per_invsestor'].sum():,.0f}")
st.metric("Unique Investors", df['investor_name'].nunique() if 'investor_name' in df.columns else "N/A")
st.metric("Unique Ecosystems", df['investee_urban_center_final'].nunique() if 'investee_urban_center_final' in df.columns else "N/A")

# Graph 1: Investments per year
st.subheader("Investments per Year")
investments_per_year = df.groupby('funding_round_year').size().reset_index(name='Count')
chart_years = alt.Chart(investments_per_year).mark_bar().encode(
    x=alt.X('funding_round_year:O', title="Year"),
    y=alt.Y('Count:Q'),
    tooltip=['funding_round_year', 'Count']
).properties(height=300)
st.altair_chart(chart_years, use_container_width=True)

# Graph 2: Top 10 ecosystems
if 'investee_urban_center_final' in df.columns:
    st.subheader("Top 10 Ecosystems by Number of Investments")
    top_ecosystems = df['investee_urban_center_final'].value_counts().nlargest(10).reset_index()
    top_ecosystems.columns = ['Ecosystem', 'Count']
    chart_ecosystems = alt.Chart(top_ecosystems).mark_bar().encode(
        x=alt.X('Count:Q'),
        y=alt.Y('Ecosystem:N', sort='-x'),
        tooltip=['Ecosystem', 'Count']
    ).properties(height=400)
    st.altair_chart(chart_ecosystems, use_container_width=True)

# Dropdown-based question mockup
st.subheader("Explore Investment Data with Predefined Questions")
questions = [
    "What are the top 5 startup ecosystems globally?",
    "Which countries have the most startup investments?",
    "What is the average investment size per funding round?",
    "Which sectors attract the most funding?",
    "Where does most AI-related investment happen?",
]
selected_question = st.selectbox("Choose a question to ask:", questions)

if selected_question:
    with st.spinner("AI is generating the answer..."):
        prompt = (
            f"You are a professional analyst specialized in global startup investments. "
            f"Based on a dataset of startup investments from 2000 to 2025, answer this question: {selected_question}"
        )
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You answer questions about startup investments."},
                {"role": "user", "content": prompt},
            ],
        )
        st.write("**Answer:**", response.choices[0].message.content)
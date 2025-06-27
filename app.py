import streamlit as st
import pandas as pd
import altair as alt
import os
from openai import OpenAI

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("metadataset_urban_centers.csv", encoding="latin1")

df = load_data()

# Filter years 2000-2025
df = df[(df['funding_round_year'] >= 2000) & (df['funding_round_year'] <= 2025)]

# Title and description
st.title("Global Startup Investment Explorer - AI-powered")
st.markdown("""
**App Description:**  
This is an AI-powered App to explore a dataset of ~630,000 startup investments from Crunchbase, carefully prepared to reflect geographic scale and startup ecosystems. At the very bottom you can ask questions to explore the database.
""")

# KPIs
st.subheader("Key Performance Indicators")
st.write("**Rows:**", df.shape[0])
st.write("**Columns:**", df.shape[1])
st.metric("Total Investment (USD)", f"${df['funding_round_amount_usd_per_invsestor'].sum():,.0f}")
st.metric("Unique Investors", df['investor_name'].nunique())
st.metric("Unique Ecosystems", df['investee_urban_center_final'].nunique())
st.metric("Average Investment per Round (USD)", f"${df['funding_round_amount_usd_per_invsestor'].mean():,.0f}")

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
st.subheader("Top 10 Ecosystems by Number of Investments")
top_ecosystems = df['investee_urban_center_final'].value_counts().nlargest(10).reset_index()
top_ecosystems.columns = ['Ecosystem', 'Count']
chart_ecosystems = alt.Chart(top_ecosystems).mark_bar().encode(
    x=alt.X('Count:Q'),
    y=alt.Y('Ecosystem:N', sort='-x'),
    tooltip=['Ecosystem', 'Count']
).properties(height=400)
st.altair_chart(chart_ecosystems, use_container_width=True)

# Graph 3: Top 10 countries
st.subheader("Top 10 Countries by Number of Investments")
top_countries = df['investee_country_code'].value_counts().nlargest(10).reset_index()
top_countries.columns = ['Country', 'Count']
chart_countries = alt.Chart(top_countries).mark_bar().encode(
    x=alt.X('Count:Q'),
    y=alt.Y('Country:N', sort='-x'),
    tooltip=['Country', 'Count']
).properties(height=400)
st.altair_chart(chart_countries, use_container_width=True)

# OpenAI Q&A with local data analysis
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.subheader("Explore Investment Data with AI")
st.markdown("Select a predefined question about global startup investments:")

questions = [
    "What are the top 5 startup ecosystems globally?",
    "Which countries have the most startup investments?",
    "How has investment volume changed over the years?",
    "What is the average investment size per funding round?",
    "Which sectors attracted the most funding?",
    "Where do most AI-related investments happen?",
    "Which investors are the most active?",
    "What are the key trends in early-stage vs. late-stage funding?",
    "Which European cities are top destinations for startup investments?",
    "How does investment activity differ between Asia and North America?",
]

selected_question = st.selectbox("Choose a question to ask:", questions)

if selected_question:
    with st.spinner("AI is generating the answer..."):
        prompt = (
            f"You are a professional analyst specialized in global startup investments. "
            f"Answer the following question with relevant insights and details, "
            f"based on a dataset of 630,000 investments from 2000 to 2025: {selected_question}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You answer questions about global startup investments."},
                {"role": "user", "content": prompt},
            ],
        )
        st.write("**Answer:**")
        st.write(response.choices[0].message.content)
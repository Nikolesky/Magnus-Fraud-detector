import streamlit as st
import pandas as pd

st.set_page_config(page_title="Magnus Fraud Detector", layout="wide")

st.title('Magnus Fraud Detector')
st.write("Criminals think in networks")
st.write("Shouldn’t your bank’s security?")

st.sidebar.title("Sidebar Title")
st.sidebar.write("Welcome to the sidebar!")
st.sidebar.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])

uploaded_file = st.file_uploader('Upload your CSV file', type='csv')

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success('File loaded successfully!')
    st.subheader('Preview of Uploaded Data')
    st.dataframe(df)
else:
    st.info('Please upload a CSV file to get started.')

if df is not None:
    # Use columns for layout
    col1, col2 = st.columns((2, 3), gap='medium')

    with col1:
        st.subheader('Top 10 Risky Accounts')
        if "risk_score" in df.columns:
            st.dataframe(df.sort_values("risk_score", ascending=False).head(10))
        else:
            st.info('No "risk_score" column found.')
        search_id = st.text_input('Search Account ID:')
        if search_id:
            search_results = df[df['account_id'] == search_id] if 'account_id' in df.columns else pd.DataFrame()
            st.write(f"Results for Account ID {search_id}:")
            st.dataframe(search_results)

    with col2:
        tab1, tab2 = st.tabs(["Charts", "Metrics"])
        with tab1:
            st.subheader('Fraud Score Distribution')
            if "risk_score" in df.columns:
                st.bar_chart(df["risk_score"])
            else:
                st.info('No "risk_score" column for chart.')
        with tab2:
            st.subheader('Model Performance Metrics')
            # Example metrics; replace with your own logic
            if "accuracy" in df.columns:
                st.metric(label='Accuracy', value=f'{df["accuracy"].mean():.2%}')
            else:
                st.write('Add your ML model metrics here.')

    with st.expander("Chatbot / Help", expanded=False):
        st.write('Coming soon: Chatbot integration for fraud queries.')

st.markdown("""
**Project Features** (you will add these, step-by-step):
- Top risk table
- Search/filter by account ID
- Fraud score chart
- Key metrics from your ML model
- Network/graph visualization (optional)
- Chatbot integration
""")

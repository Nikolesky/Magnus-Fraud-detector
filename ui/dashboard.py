import streamlit as st
import subprocess
import pandas as pd
import plotly.express as px
import os

# Page config
st.set_page_config(page_title="Magnus Fraud Detector", layout="wide")

st.title("MAGNUS FRAUD DETECTOR DASHBOARD")
st.divider()

# Sidebar
st.sidebar.header("NAVIGATION")
page = st.sidebar.radio("Select Page:", ["Home", "Search Transaction"])
st.sidebar.divider()

# Function to run C++ commands
def run_cpp_command(action, arg=None):
    cmd = ["./src/calculatingusingdatastructs", action]
    if arg:
        cmd.append(arg)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

# Load dashboard CSV
def load_dashboard_data():
    csv_path = "ui/dashboard.csv"
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'id' not in df.columns or 'risk' not in df.columns:
            return None
        return df
    except:
        return None

#HOME PAGE 
if page == "Home":
    df = load_dashboard_data()
    
    if df is not None:
        # Top 5% calculation
        topN = max(1, int(0.05 * len(df)))  # top 5%, at least 1
        top_5percent = df.nlargest(topN, 'risk')
        
        # Basic Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("Average Risk", f"{df['risk'].mean():.3f}")
        col3.metric("High Risk Count", len(df[df['risk'] > 0.7]))
        st.divider()
        
        # Top 5% Bar Chart
        st.subheader(f"Top {topN} Risky Transactions (5%)")
        fig = px.bar(
            top_5percent, x='id', y='risk',
            labels={'id': 'Transaction ID', 'risk': 'Risk Score'},
            color='risk', color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Details Table
        st.subheader("Details")
        display_df = top_5percent[['id', 'risk']].reset_index(drop=True)
        display_df.index += 1
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info("Dashboard CSV not found. Make sure C++ program has exported data.")

#SEARCH PAGE 
elif page == "Search Transaction":
    st.header("Search Transaction")
    
    id_to_search = st.text_input("Enter Transaction ID:", placeholder="e.g., U128")
    
    if st.button("Search"):
        if id_to_search:
            result = run_cpp_command("search", id_to_search)
            
            if result.returncode == 0:
                if "not found" in result.stdout.lower():
                    st.error("Transaction not found")
                else:
                    st.success("Found!")
                    st.text(result.stdout)
                    
                    # Show risk level
                    df = load_dashboard_data()
                    if df is not None and id_to_search in df['id'].values:
                        risk = df[df['id'] == id_to_search]['risk'].values[0]
                        
                        st.subheader("Risk Score")
                        st.metric("Risk Level", f"{risk:.4f}")
                        
                        if risk > 0.7:
                            st.error("HIGH RISK")
                        elif risk > 0.4:
                            st.warning("MEDIUM RISK")
                        else:
                            st.success("LOW RISK")
            else:
                st.error("Error occurred")
        else:
            st.warning("Please enter a transaction ID")

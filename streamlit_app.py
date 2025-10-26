import streamlit as st
import requests
import json
import os

# Page configuration
st.set_page_config(
    page_title="Biochar Research Assistant",
    page_icon="🌱",
    layout="wide"
)

# Get API URL from environment (for Render deployment)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Password authentication
def check_password():
    """Returns True if the password is correct."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "netzero":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.title("🔐 Authentication Required")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("Please enter the password to access the Biochar Research Assistant.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.title("🔐 Authentication Required")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😞 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Check password before showing the main app
if not check_password():
    st.stop()

# Title and description
st.title("🌱 Biochar Research Assistant")

# Main description
st.markdown("""
### About This Application

This application is a Retrieval-Augmented Generation (RAG) system that captures and organizes the collective email discussions of the Biochar Group Chat community.
It draws on conversations and shared documents from October 2008 through September 2025, turning them into a searchable, intelligent knowledge base for discovery and insight.

💡 **Ask questions about biochar research and get answers from our comprehensive knowledge base.**
""")

# Sidebar with info
with st.sidebar:
    st.header("About This RAG System")
    st.write("**Data Source:**")
    st.write("- Biochar Group communications")
    st.write("- October 2008 - September 2025")
    st.write("- 41,100+ emails and attachments")
    st.write("")
    st.write("**Technology:**")
    st.write("- Advanced embedding search")
    st.write("- Retrieval-Augmented Generation")
    st.write("- Language Model: Llama-3.2-3B-Instruct-Turbo")

# Main chat interface
st.header("💬 Ask Your Question")

# Text input for user question
user_question = st.text_area(
    placeholder="e.g., How does biochar help with carbon sequestration?",
    height=100
)

# Advanced options in expander
with st.expander("⚙️ Advanced Options"):
    max_chunks = st.slider("Maximum source chunks to retrieve", 1, 10, 5)
    include_sources = st.checkbox("Include source citations", value=True)

# Submit button
if st.button("🔍 Get Answer", type="primary"):
    if user_question.strip():
        # Show loading spinner
        with st.spinner("Searching biochar research and generating answer..."):
            try:
                # API request
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": user_question,
                        "max_chunks": max_chunks,
                        "include_sources": include_sources
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display the answer
                    st.header("📝 Answer")
                    st.write(data["response"])
                    
                    # Display sources if included
                    if include_sources and "retrieved_chunks" in data:
                        st.header("📚 Sources")
                        for i, chunk in enumerate(data["retrieved_chunks"], 1):
                            with st.expander(f"Source {i} (Relevance: {chunk['similarity']:.3f})"):
                                st.write(f"**Document:** {chunk['doc_id']}")
                                st.write(f"**Content:** {chunk['content']}")
                                st.write(f"**Tokens:** {chunk['tokens']}")
                    
                    # Display metadata
                    with st.expander("🔧 Response Details"):
                        st.write(f"**Model used:** {data.get('model_used', 'N/A')}")
                        st.write(f"**Processing time:** {data.get('processing_time_ms', 'N/A')} ms")
                        st.write(f"**Retrieved chunks:** {len(data.get('retrieved_chunks', []))}")
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to the API server. Make sure it's running on {API_URL}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The server might be busy.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question about biochar.")

# Footer
st.markdown("---")
st.markdown("*Powered by biochar research data and AI*")
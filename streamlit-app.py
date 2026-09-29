import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

st.set_page_config(
    page_title="GenAI Prototype with Data Processing",
    page_icon="🤖",
    layout="centered"
)

# Obtener credenciales del entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


# --- HELPER FUNCTIONS ---

def get_dataset_path():
    """Helper function to get dataset path adaptado a la estructura local del proyecto."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Intenta primero encontrar customer_reviews.csv dentro de la carpeta 'data' raíz de tu proyecto
    local_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    if os.path.exists(local_path):
        return local_path
    
    # Si no, busca en la ruta relativa original del módulo
    return os.path.join(current_dir, "..", "..", "data", "customer_reviews.csv")


def clean_text(text):
    """Helper function to clean text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_response(prompt_text: str, temp: float) -> str:
    """Función de consulta con fallback silencioso."""
    # 1. Intentar con OpenAI (gpt-4o)
    if openai_api_key:
        try:
            client_openai = OpenAI(api_key=openai_api_key)
            response = client_openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temp,
            )
            return response.choices[0].message.content
        except Exception:
            pass

    # 2. Fallback a DeepSeek (deepseek-chat)
    if deepseek_api_key:
        try:
            client_deepseek = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            response = client_deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temp,
            )
            return response.choices[0].message.content
        except Exception:
            pass

    return None


# --- INTERFAZ PRINCIPAL ---

st.title("🤖 Prototipo GenAI & Data Processing")
st.write("This is your GenAI-powered data processing app.")

# --- SECCIÓN 1: PROMPTS Y GENERACIÓN DE IA ---
st.subheader("💡 GenAI Assistant")

user_prompt = st.text_input(
    "Enter your prompt:",
    "Explain generative AI in one sentence."
)

temperature = st.slider(
    "Model temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = very creative"
)

if st.button("Generar Respuesta"):
    if not user_prompt.strip():
        st.warning("Por favor, ingresa una consulta válida.")
    else:
        with st.spinner("AI is working..."):
            res_text = get_response(user_prompt, temperature)
            if res_text:
                st.write(res_text)
            else:
                st.error("No fue posible obtener una respuesta en este momento.")

st.divider()

# --- SECCIÓN 2: PROCESAMIENTO DE DATOS ---
st.subheader("📊 Dataset Ingestion & Parsing")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            # Detecta automáticamente la columna de resumen independientemente de mayúsculas/minúsculas
            summary_col = next((col for col in st.session_state["df"].columns if col.upper() == "SUMMARY"), None)
            if summary_col:
                st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"][summary_col].apply(clean_text)
                st.success("Reviews parsed and cleaned!")
            else:
                st.error("Column 'SUMMARY' not found in dataset.")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    st.divider()
    product_col = next((col for col in st.session_state["df"].columns if col.upper() == "PRODUCT"), None)
    
    if product_col:
        st.subheader("🔍 Filter by Product")
        product_options = ["All Products"] + list(st.session_state["df"][product_col].dropna().unique())
        product = st.selectbox("Choose a product", product_options)
        st.subheader(f"📁 Reviews for {product}")

        if product != "All Products":
            filtered_df = st.session_state["df"][st.session_state["df"][product_col] == product]
        else:
            filtered_df = st.session_state["df"]
        st.dataframe(filtered_df)
    else:
        st.subheader("📁 Reviews Dataset")
        st.dataframe(st.session_state["df"])
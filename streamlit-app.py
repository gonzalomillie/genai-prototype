import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="GenAI Prototype", layout="centered")
st.title("🤖 Prototipo GenAI")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("No se encontró la API Key en el archivo .env")
else:
    client = OpenAI(api_key=api_key)
    prompt = st.text_input("Ingresa tu consulta:", "Explica la IA generativa en una frase")

    if st.button("Generar"):
        with st.spinner("Procesando..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error al conectar con OpenAI: {e}")
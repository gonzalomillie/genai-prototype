import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno locales
load_dotenv()

st.set_page_config(
    page_title="GenAI Prototype with Fallback",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Prototipo GenAI (OpenAI + DeepSeek Fallback)")

# Obtener credenciales del entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

prompt = st.text_input("Ingresa tu consulta:", "Explica la IA generativa en una frase")

if st.button("Generar"):
    if not prompt.strip():
        st.warning("Por favor, ingresa una consulta válida.")
    else:
        respuesta = None
        proveedor_usado = None

        # --- OPCIÓN 1: OpenAI (Intento primario) ---
        if openai_api_key:
            with st.spinner("Consultando OpenAI (gpt-4o)..."):
                try:
                    client_openai = OpenAI(api_key=openai_api_key)
                    response = client_openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                    respuesta = response.choices[0].message.content
                    proveedor_usado = "OpenAI (gpt-4o)"
                except Exception as err_openai:
                    st.warning(f"Error con OpenAI: {err_openai}. Intentando fallback con DeepSeek...")

        # --- OPCIÓN 2: DeepSeek (Fallback por error o falta de key) ---
        if respuesta is None:
            if deepseek_api_key:
                with st.spinner("Consultando DeepSeek (deepseek-chat)..."):
                    try:
                        # DeepSeek utiliza la misma interfaz compatible con el SDK de OpenAI
                        client_deepseek = OpenAI(
                            api_key=deepseek_api_key,
                            base_url="https://api.deepseek.com"
                        )
                        response = client_deepseek.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                        )
                        respuesta = response.choices[0].message.content
                        proveedor_usado = "DeepSeek (deepseek-chat)"
                    except Exception as err_deepseek:
                        st.error(f"Error con DeepSeek: {err_deepseek}")
            else:
                st.error("No se encontró la API Key de DeepSeek (DEEPSEEK_API_KEY) en las variables de entorno.")

        # --- MOSTRAR RESULTADOS ---
        if respuesta:
            st.success(f"**Respuesta recibida vía {proveedor_usado}:**")
            st.write(respuesta)
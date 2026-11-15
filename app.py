
import os
import gc
import warnings
from datetime import datetime
import tempfile
from io import BytesIO
from html import escape

import streamlit as st
import numpy as np
import torch
import torchxrayvision as xrv
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoModel, PreTrainedModel

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm


# ==========================================
# CONFIGURAÇÃO
# ==========================================

st.set_page_config(
    page_title="RADVISION IA",
    page_icon="🩻",
    layout="wide"
)

torch.set_num_threads(1)

# Limites de segurança para os arquivos enviados
TAMANHO_MAXIMO_MB = 20
DIMENSAO_MINIMA = 150
PIXELS_MAXIMOS = 40_000_000

traducoes = {
    "Atelectasis": "Atelectasia",
    "Consolidation": "Consolidação",
    "Infiltration": "Infiltração",
    "Pneumothorax": "Pneumotórax",
    "Edema": "Edema pulmonar",
    "Emphysema": "Enfisema",
    "Fibrosis": "Fibrose",
    "Effusion": "Derrame pleural",
    "Pneumonia": "Pneumonia",
    "Pleural_Thickening": "Espessamento pleural",
    "Cardiomegaly": "Cardiomegalia",
    "Nodule": "Nódulo",
    "Mass": "Massa pulmonar",
    "Hernia": "Hérnia",
    "Lung Lesion": "Lesão pulmonar",
    "Fracture": "Fratura",
    "Lung Opacity": "Opacidade pulmonar",
    "Enlarged Cardiomediastinum": "Alargamento do mediastino"
}


def traduzir_achado(nome):
    return traducoes.get(nome, nome.replace("_", " "))


# ==========================================
# VALIDAÇÃO DOS ARQUIVOS DE IMAGEM
# ==========================================

def validar_imagem(arquivo):

    if arquivo is None:
        return False, "Nenhum arquivo foi selecionado.", None

    try:

        conteudo = arquivo.getvalue()

        if not conteudo:
            return False, "O arquivo está vazio.", None

        tamanho_mb = len(conteudo) / (1024 * 1024)

        if tamanho_mb > TAMANHO_MAXIMO_MB:

            return (
                False,
                f"O arquivo possui {tamanho_mb:.1f} MB. "
                f"O limite permitido é de {TAMANHO_MAXIMO_MB} MB.",
                None
            )

        # Verifica se o conteúdo realmente corresponde a uma imagem.
        # A extensão do arquivo, sozinha, não garante isso.

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            imagem_verificacao = Image.open(
                BytesIO(conteudo)
            )

            formato = imagem_verificacao.format

            largura, altura = imagem_verificacao.size

            if formato not in ["JPEG", "PNG"]:

                return (
                    False,
                    "Formato de imagem inválido. "
                    "Envie uma imagem JPEG ou PNG.",
                    None
                )

            if largura < DIMENSAO_MINIMA or altura < DIMENSAO_MINIMA:

                return (
                    False,
                    f"A imagem possui dimensões muito pequenas "
                    f"({largura} × {altura} pixels). "
                    f"O mínimo permitido é {DIMENSAO_MINIMA} × "
                    f"{DIMENSAO_MINIMA} pixels.",
                    None
                )

            quantidade_pixels = largura * altura

            if quantidade_pixels > PIXELS_MAXIMOS:

                return (
                    False,
                    "A imagem possui dimensões excessivamente grandes "
                    "e ultrapassa o limite de segurança do sistema.",
                    None
                )

            imagem_verificacao.verify()

            # Reabre a imagem após a verificação.
            # O método verify() impede a reutilização do objeto anterior.

            imagem_final = Image.open(
                BytesIO(conteudo)
            ).convert("RGB")

            imagem_final.load()

        return (
            True,
            f"Imagem validada: {largura} × {altura} pixels.",
            imagem_final
        )

    except Image.DecompressionBombError:

        return (
            False,
            "A imagem possui dimensões excessivamente grandes "
            "e não pode ser processada com segurança.",
            None
        )

    except Image.DecompressionBombWarning:

        return (
            False,
            "A imagem ultrapassa o limite de segurança "
            "de dimensões permitido.",
            None
        )

    except Exception:

        return (
            False,
            "Não foi possível abrir a imagem. "
            "O arquivo pode estar corrompido ou não ser "
            "uma imagem JPEG ou PNG válida.",
            None
        )


# ==========================================
# MENSAGENS PERSONALIZADAS
# ==========================================

def mostrar_sucesso(mensagem):

    st.markdown(
        f"""
        <div class="radvision-success">
            <span class="success-icon">✓</span>
            <span>{escape(mensagem)}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# IDENTIDADE VISUAL
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(85,69,217,0.20), transparent 28%),
        radial-gradient(circle at 88% 18%, rgba(128,73,232,0.16), transparent 25%),
        linear-gradient(135deg, #070B1C 0%, #10152F 48%, #171536 100%);
    color: #FFFFFF;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

h1 {
    color: #FFFFFF;
    font-weight: 800;
}

h2, h3 {
    color: #C4B5FD;
}

/* CABEÇALHO */

.radvision-header {
    background: linear-gradient(
        110deg,
        rgba(37,41,79,0.98),
        rgba(54,39,105,0.94)
    );
    border: 1px solid rgba(167,139,250,0.65);
    border-left: 5px solid #A78BFA;
    border-radius: 18px;
    padding: 25px 28px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.30);
    min-height: 155px;
    box-sizing: border-box;
}

.radvision-header .radvision-kicker {
    color: #C4B5FD;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.radvision-header h1 {
    margin: 7px 0;
    font-size: 36px;
    line-height: 1.15;
    color: #FFFFFF;
    font-weight: 800;
}

.radvision-header h1 span {
    color: #B9A9FF;
}

.radvision-header p {
    color: #E9E5FF;
    margin: 0;
    font-size: 15px;
}

.radvision-header .radvision-subtitle {
    font-size: 13px;
    margin-top: 10px;
    color: #C9D4FF;
}

/* LOGO */

[data-testid="stImage"] img {
    border-radius: 18px;
    filter:
        contrast(1.12)
        brightness(1.10)
        drop-shadow(0 0 12px rgba(89,139,255,0.40))
        drop-shadow(0 0 22px rgba(139,92,246,0.30));
    transition: all 0.3s ease;
}

[data-testid="stImage"] img:hover {
    filter:
        contrast(1.15)
        brightness(1.15)
        drop-shadow(0 0 18px rgba(89,139,255,0.60))
        drop-shadow(0 0 30px rgba(139,92,246,0.40));
}

/* TÍTULOS E SEÇÕES */

.radvision-section {
    color: #D5C9FF;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(167,139,250,0.30);
}

.section-heading {
    color: #D5C9FF;
    font-size: 19px;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 12px;
}

/* ENVIO DE IMAGEM */

[data-testid="stFileUploader"] {
    background: linear-gradient(
        145deg,
        rgba(43,48,91,0.98),
        rgba(37,35,78,0.98)
    ) !important;
    border: 1px solid rgba(167,139,250,0.85) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    transition: all 0.25s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #C4B5FD !important;
    box-shadow: 0 0 22px rgba(139,120,240,0.20);
}

[data-testid="stFileUploader"] section {
    background: linear-gradient(
        145deg,
        rgba(24,29,62,0.98),
        rgba(34,35,76,0.98)
    ) !important;
    border: 2px dashed #A78BFA !important;
    border-radius: 14px !important;
    padding: 25px 14px !important;
    transition: all 0.25s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #D0BCFF !important;
    background: rgba(48,43,100,0.95) !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
    color: #D8D1FF !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: 1px solid #B9A9FF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(90deg, #705FEA, #9865F0) !important;
    border-color: #E0D9FF !important;
}

[data-testid="stFileUploader"] button p {
    color: #FFFFFF !important;
}

/* BOTÕES PRINCIPAIS */

div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(100deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    padding: 13px 20px !important;
    width: 100% !important;
    min-height: 49px !important;
    box-shadow: 0 5px 14px rgba(65,49,160,0.22);
    transition: all 0.22s ease !important;
}

div.stButton > button p,
div[data-testid="stDownloadButton"] > button p,
div[data-testid="stDownloadButton"] > button span {
    color: #FFFFFF !important;
}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(100deg, #705FEA, #9865F0) !important;
    border-color: #D0BCFF !important;
    box-shadow: 0 0 20px rgba(139,120,240,0.38);
    transform: translateY(-1px);
}

div.stButton > button:active,
div[data-testid="stDownloadButton"] > button:active {
    transform: translateY(0);
}

div.stButton > button:focus,
div[data-testid="stDownloadButton"] > button:focus {
    box-shadow: 0 0 0 3px rgba(167,139,250,0.30) !important;
    outline: none !important;
}

div[data-testid="stDownloadButton"] > button svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* CAMPOS DE TEXTO E SELEÇÃO */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stTextArea"] textarea {
    background: #20264B !important;
    color: #FFFFFF !important;
    border: 1px solid #7565D9 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus,
[data-testid="stTimeInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #B9A9FF !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.20) !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #20264B !important;
    border: 1px solid #7565D9 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}

[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stDateInput"] label,
[data-testid="stTimeInput"] label,
[data-testid="stTextArea"] label {
    color: #E0D9FF !important;
    font-weight: 700 !important;
}

[data-testid="stSelectbox"] svg,
[data-testid="stMultiSelect"] svg {
    color: #C4B5FD !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #25294F !important;
    border: 1px solid #8272E8 !important;
    border-radius: 10px !important;
}

[data-baseweb="menu"] li {
    color: #FFFFFF !important;
}

[data-baseweb="menu"] li:hover {
    background: #403777 !important;
}

/* INDICADORES NATIVOS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #25294F, #29234F) !important;
    border: 1px solid #8272E8 !important;
    border-radius: 14px !important;
    padding: 20px 18px !important;
    box-shadow: 0 5px 16px rgba(0,0,0,0.24);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] span {
    color: #E0D9FF !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p,
[data-testid="stMetricValue"] span {
    color: #FFFFFF !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
}

/* CARTÕES DE RESULTADOS */

.resultado-card {
    background: linear-gradient(145deg, #25294F, #29234F);
    border: 1px solid #8272E8;
    border-radius: 13px;
    padding: 14px 13px;
    min-height: 116px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.22);
    box-sizing: border-box;
    transition: all 0.2s ease;
}

.resultado-card:hover {
    border-color: #B9A9FF;
    box-shadow: 0 0 18px rgba(139,120,240,0.18);
    transform: translateY(-2px);
}

.resultado-label {
    color: #D8D0FF;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.7px;
    margin-bottom: 10px;
}

.resultado-valor {
    color: #FFFFFF;
    font-size: 25px;
    line-height: 1.2;
    font-weight: 800;
    overflow-wrap: anywhere;
    margin-bottom: 7px;
}

.resultado-valor.nome {
    font-size: 15px;
    line-height: 1.35;
}

.resultado-descricao {
    color: #C9C5E8;
    font-size: 10px;
    line-height: 1.4;
}

/* CAIXAS E CONTAINERS */

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(167,139,250,0.32) !important;
    border-radius: 16px !important;
    background: rgba(25,30,66,0.48) !important;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(130,114,232,0.48) !important;
    border-radius: 13px !important;
    background: rgba(32,38,75,0.55) !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: #E9E5FF !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] summary:hover {
    color: #C4B5FD !important;
    background: rgba(85,69,217,0.12) !important;
}

/* TABELAS */

[data-testid="stDataFrame"] {
    border: 1px solid rgba(130,114,232,0.55) !important;
    border-radius: 13px !important;
    overflow: hidden;
    background: rgba(32,38,75,0.65) !important;
}

/* MENSAGENS DE SUCESSO */

.radvision-success {
    display: flex;
    align-items: center;
    gap: 12px;
    background: linear-gradient(110deg, #302C5B, #29234F) !important;
    border: 1px solid #8272E8 !important;
    border-left: 5px solid #A78BFA !important;
    border-radius: 12px;
    padding: 15px 18px;
    margin: 10px 0 16px 0;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px;
    font-weight: 700;
    line-height: 1.5;
    box-shadow: 0 5px 16px rgba(0,0,0,0.20);
}

.radvision-success span {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.radvision-success .success-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    width: 25px;
    height: 25px;
    border-radius: 50%;
    background: #A78BFA;
    color: #171536 !important;
    font-size: 17px;
    font-weight: 900;
}

/* RELATÓRIO */

.relatorio-radvision {
    background: linear-gradient(145deg, #25294F, #29234F);
    border: 1px solid #8272E8;
    border-left: 5px solid #A78BFA;
    border-radius: 14px;
    padding: 25px;
    margin-top: 12px;
    margin-bottom: 20px;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 7px 22px rgba(0,0,0,0.24);
}

.relatorio-radvision .marca {
    color: #C4B5FD;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.relatorio-radvision .slogan {
    color: #E0D9FF;
    font-size: 13px;
    margin-bottom: 22px;
}

.relatorio

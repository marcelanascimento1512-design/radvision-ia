import os
import gc
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
# APARÊNCIA VISUAL
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #080D22, #151638);
    color: white;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

h1 {
    color: white;
    font-weight: 800;
}

h2, h3 {
    color: #C4B5FD;
}

div.stButton > button {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 12px 25px !important;
    width: 100%;
}

div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    width: 100% !important;
    min-height: 48px !important;
    opacity: 1 !important;
}

div[data-testid="stDownloadButton"] > button p,
div[data-testid="stDownloadButton"] > button span,
div[data-testid="stDownloadButton"] > button svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    background: #7659E8 !important;
    color: #FFFFFF !important;
}

div.stButton > button:hover {
    background: #7659E8 !important;
    color: white !important;
}

[data-testid="stFileUploader"] {
    border: 1px solid #5545D9;
    border-radius: 12px;
    padding: 12px;
}

/* INDICADORES */

[data-testid="stMetric"] {
    background: #25294F !important;
    border: 1px solid #8272E8 !important;
    border-radius: 14px !important;
    padding: 22px 18px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] span {
    color: #E0D9FF !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p,
[data-testid="stMetricValue"] span {
    color: #FFFFFF !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
}

hr {
    border-color: #5545D9;
}

/* RELATÓRIO NA TELA */

.relatorio-radvision {
    background: #25294F;
    border: 1px solid #8272E8;
    border-left: 5px solid #A78BFA;
    border-radius: 14px;
    padding: 25px;
    margin-top: 12px;
    margin-bottom: 20px;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 18px rgba(0,0,0,0.20);
}

.relatorio-radvision .marca {
    color: #C4B5FD;
    font-size: 23px;
    font-weight: 800;
}

.relatorio-radvision .slogan {
    color: #E0D9FF;
    font-size: 13px;
    margin-bottom: 22px;
}

.relatorio-radvision .secao {
    color: #C4B5FD;
    font-size: 17px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 8px;
    border-bottom: 1px solid #55558A;
    padding-bottom: 6px;
}

.relatorio-radvision .texto {
    color: #FFFFFF;
    font-size: 15px;
    line-height: 1.8;
}

.relatorio-radvision .destaque {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 800;
}

.relatorio-radvision .aviso {
    background: #34345D;
    border-radius: 8px;
    padding: 14px;
    color: #F3EFFF;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 18px;
}

/* IDENTIDADE VISUAL APRIMORADA */

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(85,69,217,0.18), transparent 28%),
        radial-gradient(circle at 88% 18%, rgba(128,73,232,0.13), transparent 25%),
        linear-gradient(135deg, #070B1C 0%, #10152F 48%, #171536 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(167,139,250,0.28) !important;
    border-radius: 16px !important;
    background: rgba(25,30,66,0.48) !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(37,41,79,0.55);
    border: 1px dashed #8B78F0;
    border-radius: 14px;
}

[data-testid="stSelectbox"] > div > div {
    background: #20264B !important;
    border: 1px solid #7565D9 !important;
    border-radius: 10px !important;
    color: white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(130,114,232,0.45);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(130,114,232,0.42) !important;
    border-radius: 12px !important;
    background: rgba(32,38,75,0.45);
}

.radvision-hero {
    background: linear-gradient(110deg, rgba(37,41,79,0.96), rgba(54,39,105,0.88));
    border: 1px solid rgba(167,139,250,0.55);
    border-left: 5px solid #A78BFA;
    border-radius: 18px;
    padding: 22px 26px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.22);
}

.radvision-kicker {
    color:#B9A9FF;
    font-size:11px;
    font-weight:800;
    letter-spacing:2px;
}

.radvision-hero h1 {
    margin:4px 0 2px 0;
    font-size:34px;
}

.radvision-hero p {
    color:#DCD8F8;
    margin:0;
}

.radvision-section {
    color:#D5C9FF;
    font-size:12px;
    font-weight:800;
    letter-spacing:1.5px;
    padding: 9px 0;
    border-bottom:1px solid rgba(167,139,250,0.25);
}

.radvision-note {
    background:rgba(48,44,91,0.72);
    border:1px solid rgba(167,139,250,0.35);
    border-radius:12px;
    padding:14px 16px;
    color:#E9E5FF;
    line-height:1.65;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODELO DE IDENTIFICAÇÃO DA INCIDÊNCIA
# ==========================================

# Este modelo não fica armazenado em cache.
# Ele será liberado após a identificação da incidência,
# antes de carregar o segundo modelo.

def carregar_modelo_incidencia():

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    modelo_view = AutoModel.from_pretrained(
        "ianpan/chest-x-ray-basic",
        trust_remote_code=True
    )

    modelo_view.eval()
    modelo_view.to("cpu")

    return modelo_view


def identificar_incidencia(arquivo):

    modelo_view = carregar_modelo_incidencia()

    imagem_pil = Image.open(
        BytesIO(arquivo.getvalue())
    ).convert("L")

    imagem_array = np.array(imagem_pil)

    imagem_processada = modelo_view.preprocess(
        imagem_array
    )

    entrada_view = torch.from_numpy(
        np.asarray(imagem_processada)
    ).unsqueeze(0).unsqueeze(0).float()

    with torch.inference_mode():
        saida_view = modelo_view(entrada_view)

    pontuacoes_view = torch.softmax(
        saida_view["view"],
        dim=1
    )[0]

    classes_view = ["AP", "PA", "Perfil"]

    indice = int(torch.argmax(pontuacoes_view).item())

    incidencia = classes_view[indice]

    escore = float(
        pontuacoes_view[indice].item()
    ) * 100

    # LIBERAÇÃO DA MEMÓRIA DO PRIMEIRO MODELO

    del saida_view
    del pontuacoes_view
    del entrada_view
    del imagem_processada
    del imagem_array
    del imagem_pil
    del modelo_view

    gc.collect()

    return incidencia, escore


# ==========================================
# MODELO DE ANÁLISE RADIOGRÁFICA
# ==========================================

@st.cache_resource
def carregar_modelo():

    modelo = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )

    modelo.eval()

    return modelo


# ==========================================
# PREPARAÇÃO DA RADIOGRAFIA
# ==========================================

def preparar_imagem(arquivo):

    imagem_pil = Image.open(
        BytesIO(arquivo.getvalue())
    ).convert("L")

    arquivo_temporario = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    caminho = arquivo_temporario.name
    arquivo_temporario.close()

    try:
        imagem_pil.save(caminho)
        imagem = xrv.utils.load_image(caminho)

    finally:
        if os.path.exists(caminho):
            os.remove(caminho)

    altura, largura = imagem.shape[-2:]
    lado = min(altura, largura)

    topo = (altura - lado) // 2
    esquerda = (largura - lado) // 2

    imagem = imagem[
        ...,
        topo:topo + lado,
        esquerda:esquerda + lado
    ]

    imagem = xrv.datasets.XRayResizer(224)(imagem)

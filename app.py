
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

.radvision-hero {
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
}

.radvision-kicker {
    color: #C4B5FD;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.radvision-hero h1 {
    margin: 7px 0 3px 0;
    font-size: 36px;
}

.radvision-hero p {
    color: #E9E5FF;
    margin: 0;
}

/* LOGO */

[data-testid="stImage"] img {
    border-radius: 14px;
    filter: contrast(1.08) brightness(1.06);
}

/* TÍTULOS */

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
        rgba(43,48,91,0.96),
        rgba(37,35,78,0.96)
    ) !important;
    border: 2px solid #8B78F0 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

[data-testid="stFileUploader"] section {
    background: rgba(24,29,62,0.95) !important;
    border: 2px dashed #A78BFA !important;
    border-radius: 14px !important;
    padding: 22px 12px !important;
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

[data-testid="stFileUploader"] small {
    color: #D8D1FF !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: 1px solid #B9A9FF !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] button p {
    color: #FFFFFF !important;
}

/* BOTÕES */

div.stButton > button {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: 1px solid #A78BFA !important;
    border-radius: 11px !important;
    font-weight: 800 !important;
    padding: 13px 20px !important;
    width: 100%;
    min-height: 48px;
    transition: 0.2s ease;
}

div.stButton > button p {
    color: #FFFFFF !important;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #705FEA, #9865F0) !important;
    border-color: #D0BCFF !important;
    box-shadow: 0 0 18px rgba(139,120,240,0.35);
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

/* INDICADORES NATIVOS */

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
    font-size: 36px !important;
    font-weight: 800 !important;
    line-height: 1.3 !important;
}

/* NOVOS CARTÕES DE RESULTADO */

.result-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 16px 0 12px 0;
}

.result-card {
    background: linear-gradient(145deg, #25294F, #30275F);
    border: 1px solid rgba(167,139,250,0.65);
    border-top: 4px solid #9B83F5;
    border-radius: 15px;
    padding: 19px 17px;
    min-height: 130px;
    box-shadow: 0 7px 20px rgba(0,0,0,0.22);
    overflow-wrap: anywhere;
}

.result-card:nth-child(2) {
    border-top-color: #7C8CFB;
}

.result-card:nth-child(3) {
    border-top-color: #C084FC;
}

.result-label {
    color: #D6CCFF;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.result-value {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.2;
}

.result-detail {
    color: #C9C4E8;
    font-size: 11px;
    margin-top: 9px;
    line-height: 1.5;
}

/* CAIXAS */

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(167,139,250,0.28) !important;
    border-radius: 16px !important;
    background: rgba(25,30,66,0.48) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #20264B !important;
    border: 1px solid #7565D9 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
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

.relatorio-radvision .secao {
    color: #C4B5FD;
    font-size: 16px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 9px;
    border-bottom: 1px solid #55558A;
    padding-bottom: 7px;
}

.relatorio-radvision .texto {
    color: #FFFFFF;
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 5px;
}

.relatorio-radvision .aviso {
    background: #34345D;
    border: 1px solid rgba(192,132,252,0.35);
    border-radius: 9px;
    padding: 15px;
    color: #F3EFFF;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 20px;
}

/* AVISOS */

.radvision-note {
    background: rgba(48,44,91,0.72);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 12px;
    padding: 14px 16px;
    color: #E9E5FF;
    line-height: 1.65;
}

/* GRÁFICOS */

.chart-shell {
    background: rgba(25,30,66,0.55);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 16px;
    padding: 12px;
    margin: 8px 0 18px 0;
}

@media (max-width: 700px) {
    .result-grid {
        grid-template-columns: 1fr;
    }

    .radvision-hero h1 {
        font-size: 27px;
    }
}

hr {
    border-color: #5545D9;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# FUNÇÕES AUXILIARES DOS GRÁFICOS
# ==========================================

def configurar_grafico(fig, ax):
    fig.patch.set_facecolor("#171B3A")
    ax.set_facecolor("#20264B")

    ax.tick_params(colors="#E9E5FF", labelsize=10)
    ax.xaxis.label.set_color("#E9E5FF")
    ax.yaxis.label.set_color("#E9E5FF")
    ax.title.set_color("#FFFFFF")

    for borda in ax.spines.values():
        borda.set_color("#55558A")

    ax.grid(
        axis="x",
        color="#7770B8",
        alpha=0.22,
        linestyle="--"
    )


# ==========================================
# MODELO DE IDENTIFICAÇÃO DA INCIDÊNCIA
# ==========================================

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

    tensor = torch.from_numpy(
        imagem
    ).unsqueeze(0).float()

    return imagem, tensor


# ==========================================
# RELATÓRIO ESTRUTURADO
# ==========================================

def construir_relatorio(
    incidencia,
    escore_incidencia,
    principais
):

    linhas = []

    linhas.append("RESUMO DA ANÁLISE COMPUTACIONAL")

    linhas.append(
        "O sistema realizou uma análise experimental "
        "da imagem radiográfica enviada."
    )

    linhas.append("")
    linhas.append("1. IDENTIFICAÇÃO DA INCIDÊNCIA")

    linhas.append(
        f"Incidência sugerida pelo classificador: {incidencia}"
    )

    linhas.append(
        f"Escore do classificador: {escore_incidencia:.2f}%"
    )

    linhas.append(
        "A classificação automática deve ser conferida "
        "por profissional habilitado."
    )

    linhas.append("")
    linhas.append("2. PRINCIPAIS ESCORES DO MODELO")

    if principais:
        for nome, valor in principais:
            linhas.append(
                f"- {traduzir_achado(nome)}: {float(valor):.4f}"
            )
    else:
        linhas.append("Não foram disponibilizados resultados.")

    linhas.append("")
    linhas.append("3. OBSERVAÇÕES TÉCNICAS")

    linhas.append(
        "Os resultados correspondem às saídas de um modelo "
        "computacional aplicado à imagem."
    )

    linhas.append(
        "Os escores não representam probabilidades diagnósticas "
        "e não confirmam a presença ou ausência de doenças."
    )

    linhas.append(
        "O mapa de sensibilidade representa uma visualização "
        "experimental da resposta do modelo, não sendo uma "
        "confirmação da localização de alterações."
    )

    linhas.append("")
    linhas.append("4. CONCLUSÃO EXPERIMENTAL")

    linhas.append(
        "A análise computacional foi concluída e os resultados "
        "foram organizados para fins acadêmicos e demonstrativos."
    )

    linhas.append(
        "A interpretação da radiografia deve considerar a "
        "qualidade da imagem, o contexto clínico e a avaliação "
        "de um profissional habilitado."
    )

    return "\n".join(linhas)


# ==========================================
# GERAÇÃO DO PDF
# ==========================================

def gerar_pdf(incidencia, escore, principais, relatorio):

    pdf_buffer = BytesIO()
    estilos = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloRADVISION",
        parent=estilos["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5545C7"),
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        spaceAfter=8
    )

    slogan = ParagraphStyle(
        "SloganRADVISION",
        parent=estilos["Normal"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        fontSize=10
    )

    subtitulo = ParagraphStyle(
        "SubtituloRADVISION",
        parent=estilos["Heading2"],
        textColor=colors.HexColor("#5545C7"),
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=10
    )

    texto = ParagraphStyle(
        "TextoRADVISION",
        parent=estilos["BodyText"],
        textColor=colors.HexColor("#222222"),
        fontSize=10,
        leading=15,
        spaceAfter=6
    )

    elementos = [
        Paragraph("RADVISION IA", titulo),
        Paragraph(
            "Tecnologia e precisão em cada imagem.",
            slogan
        ),
        Spacer(1, 25),
        Paragraph("RELATÓRIO EXPERIMENTAL ESTRUTURADO", subtitulo),
        Paragraph(
            f"<b>Incidência sugerida:</b> {escape(incidencia)}",
            texto
        ),
        Paragraph(
            f"<b>Escore do classificador:</b> {escore:.2f}%",
            texto
        ),
        Spacer(1, 12),
        Paragraph("PRINCIPAIS ESCORES DO MODELO", subtitulo)
    ]

    dados_pdf = [
        ["Achado identificado", "Escore do modelo"]
    ]

    for nome, valor in principais:
        dados_pdf.append([
            escape(traduzir_achado(nome)),
            f"{float(valor):.4f}"
        ])

    tabela_pdf = Table(
        dados_pdf,
        colWidths=[10 * cm, 6 * cm],
        repeatRows=1
    )

    tabela_pdf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5545C7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#222222")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F1FA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 9)
    ]))

    elementos.extend([
        Spacer(1, 10),
        tabela_pdf,
        Spacer(1, 20),
        Paragraph("RELATÓRIO DESCRITIVO", subtitulo)
    ])

    secoes = relatorio.split("\n\n")

    for secao in secoes:

        texto_secao = escape(secao).replace("\n", "<br/>")

        if texto_secao.strip():
            elementos.append(
                Paragraph(texto_secao, texto)
            )
            elementos.append(Spacer(1, 5))

    elementos.extend([
        Spacer(1, 15),
        Paragraph("AVISO IMPORTANTE", subtitulo),
        Paragraph(
            "Este sistema possui finalidade acadêmica e experimental. "
            "Os escores não representam probabilidades diagnósticas. "
            "A classificação da incidência não garante acerto. "
            "O mapa de sensibilidade não confirma a localização de "
            "lesões. O RADVISION IA não deve ser utilizado para "
            "diagnóstico ou decisões clínicas e não substitui a "
            "avaliação de um profissional habilitado.",
            texto
        )
    ])

    def rodape(canvas, documento):

        canvas.saveState()
        largura, altura = A4

        canvas.setStrokeColor(
            colors.HexColor("#5545C7")
        )

        canvas.line(
            2 * cm, 1.7 * cm,
            largura - 2 * cm, 1.7 * cm
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(
            colors.HexColor("#666666")
        )

        canvas.drawString(
            2 * cm, 1.2 * cm,
            "RADVISION IA | Projeto acadêmico"
        )

        canvas.drawRightString(
            largura - 2 * cm, 1.2 * cm,
            f"Página {documento.page}"
        )

        canvas.restoreState()

    documento = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.3 * cm,
        title="Relatório experimental RADVISION IA",
        author="RADVISION IA"
    )

    documento.build(
        elementos,
        onFirstPage=rodape,
        onLaterPages=rodape
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ==========================================
# ANÁLISE DA RADIOGRAFIA
# ==========================================

def analisar_radiografia(arquivo):

    incidencia, escore_incidencia = identificar_incidencia(
        arquivo
    )

    modelo = carregar_modelo()

    imagem, entrada = preparar_imagem(arquivo)

    entrada = entrada.to("cpu")

    with torch.inference_mode():
        resultado = modelo(entrada)

    pontuacoes = resultado[0].cpu().numpy()

    analise = sorted(
        zip(modelo.pathologies, pontuacoes),
        key=lambda x: x[1],
        reverse=True
    )

    principais = analise[:10]

    tabela_resultados = [
        {
            "Achado identificado": traduzir_achado(nome),
            "Escore do modelo": round(float(valor), 4)
        }
        for nome, valor in analise
    ]

    # GRÁFICO PRINCIPAL

    fig, ax = plt.subplots(figsize=(12, 7))

    nomes = [
        traduzir_achado(item[0])
        for item in principais
    ]

    valores = [
        float(item[1])
        for item in principais
    ]

    cores_barras = [
        "#A78BFA",
        "#9381F5",
        "#8177E8",
        "#747FE5",
        "#6D8DE8",
        "#718DE0",
        "#7D83D9",
        "#8D7BD3",
        "#9A76D1",
        "#A77BCB"
    ]

    ax.barh(
        nomes[::-1],
        valores[::-1],
        color=cores_barras[:len(valores)][::-1],
        height=0.66,
        edgecolor="#D8CCFF",
        linewidth=0.5
    )

    ax.set_title(
        "RADVISION IA | Principais resultados",
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Escore do modelo",
        fontsize=11,
        labelpad=10
    )

    ax.set_ylabel("")

    configurar_grafico(fig, ax)

    ax.grid(axis="y", visible=False)

    fig.tight_layout(pad=2)

    # MAPA DE SENSIBILIDADE

    entrada_mapa = entrada.detach().clone()
    entrada_mapa.requires_grad_(True)

    modelo.zero_grad()

    saida = modelo(entrada_mapa)

    indice_maior = int(np.argmax(pontuacoes))

    saida[0, indice_maior].backward()

    gradientes = entrada_mapa.grad.abs()

    mapa = gradientes[0].max(
        dim=0
    )[0].detach().numpy()

    mapa = (mapa - mapa.min()) / (
        mapa.max() - mapa.min() + 1e-8
    )

    fig_mapa, ax_mapa = plt.subplots(figsize=(8, 8))

    fig_mapa.patch.set_facecolor("#171B3A")
    ax_mapa.set_facecolor("#171B3A")

    ax_mapa.imshow(
        imagem[0],
        cmap="gray"
    )

    ax_mapa.imshow(
        mapa,
        cmap="magma",
        alpha=0.48
    )

    ax_mapa.set_title(
        "Mapa experimental de sensibilidade",
        color="#FFFFFF",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax_mapa.axis("off")

    fig_mapa.tight_layout(pad=1.5)

    relatorio = construir_relatorio(
        incidencia,
        escore_incidencia,
        principais
    )

    pdf = gerar_pdf(
        incidencia,
        escore_incidencia,
        principais,
        relatorio
    )

    del entrada_mapa
    del gradientes
    del resultado
    del saida
    del entrada
    del modelo

    gc.collect()

    return (
        incidencia,
        escore_incidencia,
        tabela_resultados,
        fig,
        fig_mapa,
        relatorio,
        pdf
    )


# ==========================================
# CABEÇALHO
# ==========================================

coluna_logo, coluna_titulo = st.columns(
    [1, 3],
    vertical_alignment="center"
)

with coluna_logo:

    if os.path.exists("IMG_5787.jpeg"):
        st.image("IMG_5787.jpeg", width=210)
    else:
        st.markdown("## 🩻")

with coluna_titulo:

    st.markdown("""
    <div class="radvision-hero">
        <div class="radvision-kicker">
            PLATAFORMA DE ANÁLISE RADIOGRÁFICA
        </div>
        <h1>RADVISION <span style="color:#B9A9FF;">IA</span></h1>
        <p><b>Tecnologia e precisão em cada imagem.</b></p>
        <p style="font-size:13px; margin-top:8px;">
            Análise experimental de radiografias de tórax
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

if "historico_analises" not in st.session_state:
    st.session_state.historico_analises = []

st.markdown("## ENVIO DA RADIOGRAFIA")

st.markdown("""
<div class="radvision-note">
    <b>ÁREA DE ENVIO</b><br>
    Selecione uma radiografia de tórax nos formatos JPG, JPEG ou PNG
    para iniciar a análise computacional experimental.
</div>
""", unsafe_allow_html=True)

st.write("")

coluna_imagem, coluna_resultado = st.columns(
    [1, 1.4],
    gap="large"
)

with coluna_imagem:

    st.markdown("### 📤 Adicionar radiografia")

    arquivo = st.file_uploader(
        "Clique abaixo para selecionar sua imagem",
        type=["jpg", "jpeg", "png"],
        help="Formatos aceitos: JPG, JPEG e PNG"
    )

    if arquivo is not None:

        st.image(
            arquivo,
            caption="Radiografia enviada",
            use_container_width=True
        )

        st.success("Imagem carregada com sucesso.")

    analisar = st.button(
        "🩻 ANALISAR RADIOGRAFIA",
        use_container_width=True
    )


with coluna_resultado:

    st.markdown("## RESULTADOS DA ANÁLISE")

    if analisar:

        if arquivo is None:

            st.warning(
                "Envie uma radiografia primeiro."
            )

        else:

            try:

                with st.spinner(
                    "Identificando a incidência e realizando "
                    "a análise computacional..."
                ):

                    (
                        incidencia,
                        escore_incidencia,
                        tabela,
                        grafico,
                        mapa,
                        relatorio,
                        pdf
                    ) = analisar_radiografia(arquivo)

                registro = {
                    "data_hora": datetime.now().strftime("%d/%m/%Y às %H:%M"),
                    "arquivo": arquivo.name,
                    "incidencia": incidencia,
                    "escore_incidencia": escore_incidencia,
                    "tabela": tabela,
                    "relatorio": relatorio,
                    "pdf": pdf,
                }

                st.session_state.historico_analises.insert(0, registro)

                st.success(
                    "Análise computacional concluída!"
                )

                # CARTÕES DE DESTAQUE

                maior_achado = tabela[0]["Achado identificado"]
                maior_escore = tabela[0]["Escore do modelo"]

                st.markdown("""
                <div class="section-heading">
                    PAINEL DE RESULTADOS
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="result-grid">
                        <div class="result-card">
                            <div class="result-label">
                                Incidência sugerida
                            </div>
                            <div class="result-value">
                                {escape(incidencia)}
                            </div>
                            <div class="result-detail">
                                Classificação automática da imagem
                            </div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">
                                Escore do classificador
                            </div>
                            <div class="result-value">
                                {escore_incidencia:.2f}%
                            </div>
                            <div class="result-detail">
                                Pontuação da classificação de incidência
                            </div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">
                                Maior escore do modelo
                            </div>
                            <div class="result-value" style="font-size:21px;">
                                {escape(maior_achado)}
                            </div>
                            <div class="result-detail">
                                Escore numérico: {maior_escore:.4f}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "Os escores são saídas dos modelos computacionais. "
                    "Não representam probabilidades diagnósticas nem "
                    "garantia de acerto."
                )

                st.markdown("### Achados identificados")

                st.dataframe(
                    tabela,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Gráfico de resultados")

                st.markdown(
                    '<div class="chart-shell">',
                    unsafe_allow_html=True
                )

                st.pyplot(
                    grafico,
                    use_container_width=True
                )

                st.markdown("</div>", unsafe_allow_html=True)

                plt.close(grafico)

                st.markdown("### Mapa de sensibilidade")

                st.markdown(
                    """
                    <div class="radvision-note">
                        Visualização experimental da resposta do modelo
                        aos diferentes pontos da imagem. As regiões destacadas
                        não representam localização confirmada de lesões.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                mapa_col1, mapa_col2, mapa_col3 = st.columns(
                    [0.25, 1.5, 0.25]
                )

                with mapa_col2:

                    st.markdown(
                        '<div class="chart-shell">',
                        unsafe_allow_html=True
                    )

                    st.pyplot(
                        mapa,
                        use_container_width=True
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                plt.close(mapa)

                st.markdown("### Relatório experimental estruturado")

                secoes = relatorio.split("\n\n")

                html_secoes = ""

                for secao in secoes:

                    linhas = secao.split("\n")

                    titulo_secao = linhas[0]
                    conteudo = linhas[1:]

                    html_secoes += (
                        '<div class="secao">'
                        + escape(titulo_secao)
                        + '</div>'
                    )

                    for linha in conteudo:

                        if linha.strip():
                            html_secoes += (
                                '<div class="texto">'
                                + escape(linha)
                                + '</div>'
                            )

                st.html(
                    f"""
                    <div class="relatorio-radvision">
                        <div class="marca">RADVISION IA</div>

                        <div class="slogan">
                            Tecnologia e precisão em cada imagem.
                        </div>

                        {html_secoes}

                        <div class="aviso">
                            <b>AVISO:</b> ferramenta experimental
                            para fins acadêmicos. Os resultados
                            não representam diagnóstico e não
                            substituem a avaliação profissional.
                        </div>
                    </div>
                    """
                )

                st.download_button(
                    label="📄 BAIXAR RELATÓRIO EM PDF",
                    data=pdf,
                    file_name="relatorio_radvision_ia.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except Exception as erro:

                st.error(
                    f"Não foi possível concluir a análise: {erro}"
                )


# ==========================================
# HISTÓRICO DE ANÁLISES
# ==========================================

st.divider()

st.markdown("## HISTÓRICO DE ANÁLISES")

st.caption(
    "As análises ficam disponíveis enquanto esta sessão do "
    "aplicativo estiver aberta. Ao reiniciar a sessão, o histórico será apagado."
)


# ==========================================
# COMPARAÇÃO ENTRE DUAS ANÁLISES
# ==========================================

if len(st.session_state.historico_analises) >= 2:

    st.markdown("### COMPARAÇÃO ENTRE ANÁLISES")

    st.markdown(
        '<div class="radvision-note">Selecione duas análises armazenadas para comparar '
        'os escores produzidos pelo modelo. Esta comparação é apenas computacional '
        'e não determina melhora, piora ou evolução clínica.</div>',
        unsafe_allow_html=True
    )

    opcoes_comparacao = list(
        range(len(st.session_state.historico_analises))
    )

    def rotulo_analise(indice_opcao):

        item_opcao = st.session_state.historico_analises[
            indice_opcao
        ]

        return (
            f"{item_opcao['data_hora']} | "
            f"{item_opcao['arquivo']} | "
            f"{item_opcao['incidencia']}"
        )

    col_comp_a, col_comp_b = st.columns(2)

    with col_comp_a:

        indice_a = st.selectbox(
            "Primeira análise",
            options=opcoes_comparacao,
            index=1,
            format_func=rotulo_analise,
            key="comparacao_analise_a"
        )

    opcoes_b = [
        i for i in opcoes_comparacao
        if i != indice_a
    ]

    with col_comp_b:

        indice_b = st.selectbox(
            "Segunda análise",
            options=opcoes_b,
            index=0,
            format_func=rotulo_analise,
            key="comparacao_analise_b"
        )

    analise_a = st.session_state.historico_analises[indice_a]
    analise_b = st.session_state.historico_analises[indice_b]

    met_a, met_b = st.columns(2)

    with met_a:

        st.metric(
            "Incidência — análise A",
            analise_a["incidencia"]
        )

        st.metric(
            "Escore de incidência — A",
            f"{analise_a['escore_incidencia']:.2f}%"
        )

    with met_b:

        st.metric(
            "Incidência — análise B",
            analise_b["incidencia"]
        )

        st.metric(
            "Escore de incidência — B",
            f"{analise_b['escore_incidencia']:.2f}%"
        )

    mapa_a = {
        linha["Achado identificado"]: float(
            linha["Escore do modelo"]
        )
        for linha in analise_a["tabela"]
    }

    mapa_b = {
        linha["Achado identificado"]: float(
            linha["Escore do modelo"]
        )
        for linha in analise_b["tabela"]
    }

    nomes_comuns = sorted(
        set(mapa_a) | set(mapa_b)
    )

    tabela_comparacao = []

    for nome_achado in nomes_comuns:

        valor_a = mapa_a.get(nome_achado, 0.0)
        valor_b = mapa_b.get(nome_achado, 0.0)

        tabela_comparacao.append({
            "Achado identificado": nome_achado,
            "Escore A": round(valor_a, 4),
            "Escore B": round(valor_b, 4),
            "Diferença (B - A)": round(
                valor_b - valor_a,
                4
            )
        })

    tabela_comparacao.sort(
        key=lambda linha: abs(
            linha["Diferença (B - A)"]
        ),
        reverse=True
    )

    st.markdown("#### Comparação dos escores por achado")

    st.dataframe(
        tabela_comparacao,
        use_container_width=True,
        hide_index=True
    )

    maiores_diferencas = tabela_comparacao[:10]

    nomes_grafico = [
        linha["Achado identificado"]
        for linha in maiores_diferencas
    ][::-1]

    valores_a = [
        linha["Escore A"]
        for linha in maiores_diferencas
    ][::-1]

    valores_b = [
        linha["Escore B"]
        for linha in maiores_diferencas
    ][::-1]

    fig_comp, ax_comp = plt.subplots(figsize=(12, 7))

    posicoes = np.arange(len(nomes_grafico))
    altura_barra = 0.36

    ax_comp.barh(
        posicoes - altura_barra / 2,
        valores_a,
        altura_barra,
        label="Análise A",
        color="#8B78F0",
        edgecolor="#D8CCFF",
        linewidth=0.5
    )

    ax_comp.barh(
        posicoes + altura_barra / 2,
        valores_b,
        altura_barra,
        label="Análise B",
        color="#5C9BEB",
        edgecolor="#C5E0FF",
        linewidth=0.5
    )

    ax_comp.set_yticks(posicoes)
    ax_comp.set_yticklabels(nomes_grafico)

    ax_comp.set_xlabel("Escore do modelo")

    ax_comp.set_title(
        "Comparação dos escores — 10 maiores diferenças",
        fontsize=15,
        fontweight="bold",
        pad=16
    )

    configurar_grafico(fig_comp, ax_comp)

    ax_comp.grid(axis="y", visible=False)

    legenda = ax_comp.legend(
        facecolor="#25294F",
        edgecolor="#8272E8",
        labelcolor="#FFFFFF"
    )

    fig_comp.tight_layout()

    st.markdown(
        '<div class="chart-shell">',
        unsafe_allow_html=True
    )

    st.pyplot(
        fig_comp,
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    plt.close(fig_comp)

    st.caption(
        "A diferença (B - A) é uma subtração entre escores do modelo. "
        "Valores positivos indicam escore numericamente maior na análise B; "
        "valores negativos indicam escore numericamente menor. Isso não equivale "
        "a aumento ou redução de doença. As imagens podem ter diferenças de "
        "posicionamento, qualidade, técnica e processamento."
    )


elif len(st.session_state.historico_analises) == 1:

    st.info(
        "Faça pelo menos mais uma análise nesta sessão "
        "para liberar a comparação."
    )


# ==========================================
# EXIBIÇÃO DO HISTÓRICO
# ==========================================

if st.session_state.historico_analises:

    coluna_historico, coluna_limpar = st.columns([3, 1])

    with coluna_historico:

        st.markdown(
            f"**Análises realizadas nesta sessão: "
            f"{len(st.session_state.historico_analises)}**"
        )

    with coluna_limpar:

        if st.button(
            "LIMPAR HISTÓRICO",
            use_container_width=True
        ):

            st.session_state.historico_analises = []
            st.rerun()

    for indice, item in enumerate(
        st.session_state.historico_analises
    ):

        titulo_item = (
            f"{item['data_hora']} | {item['arquivo']} | "
            f"Incidência: {item['incidencia']}"
        )

        with st.expander(titulo_item):

            coluna_inc_hist, coluna_escore_hist = st.columns(2)

            with coluna_inc_hist:

                st.metric(
                    "Incidência sugerida",
                    item["incidencia"]
                )

            with coluna_escore_hist:

                st.metric(
                    "Escore do classificador",
                    f"{item['escore_incidencia']:.2f}%"
                )

            st.markdown("### Achados identificados")

            st.dataframe(
                item["tabela"],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Relatório experimental")

            st.text(item["relatorio"])

            st.download_button(
                label="📄 BAIXAR ESTE RELATÓRIO EM PDF",
                data=item["pdf"],
                file_name=f"relatorio_radvision_historico_{indice + 1}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"historico_pdf_{indice}"
            )

else:

    st.info(
        "Nenhuma análise foi armazenada nesta sessão. "
        "Envie uma radiografia e clique em ANALISAR RADIOGRAFIA."
    )


# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:18px;">
    <h3>RADVISION IA | Projeto acadêmico</h3>
    <p>
        Sistema experimental de análise de radiografias
        de tórax.
    </p>
    <p>
        Os resultados não substituem a avaliação
        de um profissional habilitado.
    </p>
    <p style="color:#A78BFA; font-size:12px;">
        Tecnologia e precisão em cada imagem.
    </p>
</div>
""", unsafe_allow_html=True)

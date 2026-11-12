
import os
import tempfile
from io import BytesIO

import streamlit as st
import numpy as np
import torch
import torchxrayvision as xrv
import matplotlib.pyplot as plt

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

/* Botão principal */
div.stButton > button {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 12px 25px !important;
    width: 100%;
}

/* Botão de download: texto sempre visível */
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
    border-color: #C4B5FD !important;
}

/* Garante contraste também no foco */
div[data-testid="stDownloadButton"] > button:focus,
div[data-testid="stDownloadButton"] > button:active {
    color: #FFFFFF !important;
    background: #654BD6 !important;
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

[data-testid="stMetric"] {
    background: #202044;
    padding: 15px;
    border-radius: 12px;
}

hr {
    border-color: #5545D9;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# CARREGAMENTO DO MODELO
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

    imagem_pil = Image.open(arquivo).convert("L")

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

    tensor = torch.from_numpy(imagem).unsqueeze(0).float()

    return imagem, tensor


# ==========================================
# GERAÇÃO DO PDF
# ==========================================

def gerar_pdf(incidencia, principais):

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
        fontName="Helvetica",
        fontSize=10,
        leading=14
    )

    subtitulo = ParagraphStyle(
        "SubtituloRADVISION",
        parent=estilos["Heading2"],
        alignment=TA_LEFT,
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
        alignment=TA_LEFT,
        textColor=colors.HexColor("#222222"),
        fontName="Helvetica",
        fontSize=10,
        leading=15
    )

    aviso = ParagraphStyle(
        "AvisoRADVISION",
        parent=texto,
        textColor=colors.HexColor("#333333"),
        fontSize=9,
        leading=14,
        spaceBefore=12
    )

    elementos = []

    elementos.append(Paragraph("RADVISION IA", titulo))

    elementos.append(
        Paragraph(
            "Tecnologia e precisão em cada imagem.",
            slogan
        )
    )

    elementos.append(Spacer(1, 25))

    elementos.append(
        Paragraph("RELATÓRIO EXPERIMENTAL", subtitulo)
    )

    elementos.append(
        Paragraph(
            f"<b>Incidência informada:</b> {incidencia}",
            texto
        )
    )

    elementos.append(Spacer(1, 15))

    elementos.append(
        Paragraph(
            "PRINCIPAIS ACHADOS IDENTIFICADOS",
            subtitulo
        )
    )

    dados_pdf = [
        ["Achado identificado", "Escore do modelo"]
    ]

    for nome, valor in principais:
        dados_pdf.append([
            traduzir_achado(nome),
            f"{float(valor):.4f}"
        ])

    tabela_pdf = Table(
        dados_pdf,
        colWidths=[10 * cm, 6 * cm],
        repeatRows=1
    )

    tabela_pdf.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#5545C7")
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold"
        ),
        (
            "FONTNAME",
            (0, 1),
            (-1, -1),
            "Helvetica"
        ),
        (
            "TEXTCOLOR",
            (0, 1),
            (-1, -1),
            colors.HexColor("#222222")
        ),
        (
            "BACKGROUND",
            (0, 1),
            (-1, -1),
            colors.HexColor("#F3F1FA")
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.HexColor("#CCCCCC")
        ),
        (
            "ALIGN",
            (1, 0),
            (1, -1),
            "CENTER"
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            9
        )
    ]))

    elementos.append(tabela_pdf)

    elementos.append(Spacer(1, 25))

    elementos.append(
        Paragraph(
            "<b>OBSERVAÇÃO IMPORTANTE</b>",
            subtitulo
        )
    )

    elementos.append(
        Paragraph(
            "Os valores apresentados são escores experimentais "
            "produzidos pelo modelo computacional e não representam "
            "probabilidades diagnósticas. O mapa de sensibilidade "
            "não confirma a localização de lesões. Este sistema "
            "possui finalidade acadêmica e experimental, não devendo "
            "ser utilizado para diagnóstico ou decisões clínicas. "
            "Os resultados não substituem a avaliação de um "
            "profissional habilitado.",
            aviso
        )
    )

    def rodape(canvas, documento):
        canvas.saveState()

        largura, altura = A4

        canvas.setStrokeColor(colors.HexColor("#5545C7"))
        canvas.setLineWidth(1)

        canvas.line(
            2 * cm,
            1.7 * cm,
            largura - 2 * cm,
            1.7 * cm
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))

        canvas.drawString(
            2 * cm,
            1.2 * cm,
            "RADVISION IA | Projeto acadêmico"
        )

        canvas.drawRightString(
            largura - 2 * cm,
            1.2 * cm,
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

def analisar_radiografia(arquivo, incidencia):

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

    # GRÁFICO

    fig, ax = plt.subplots(figsize=(10, 6))

    nomes = [
        traduzir_achado(item[0])
        for item in principais
    ]

    valores = [
        float(item[1])
        for item in principais
    ]

    ax.barh(nomes[::-1], valores[::-1])

    ax.set_title(
        "RADVISION IA | Principais resultados",
        fontsize=15
    )

    ax.set_xlabel("Escore do modelo")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()

    # MAPA DE SENSIBILIDADE

    entrada_mapa = entrada.detach().clone()
    entrada_mapa.requires_grad_(True)

    modelo.zero_grad()

    saida = modelo(entrada_mapa)

    indice_maior = int(np.argmax(pontuacoes))

    saida[0, indice_maior].backward()

    gradientes = entrada_mapa.grad.abs()

    mapa = gradientes[0].max(dim=0)[0].detach().numpy()

    mapa = (mapa - mapa.min()) / (
        mapa.max() - mapa.min() + 1e-8
    )

    fig_mapa, ax_mapa = plt.subplots(figsize=(7, 7))

    ax_mapa.imshow(imagem[0], cmap="gray")
    ax_mapa.imshow(mapa, cmap="jet", alpha=0.4)

    ax_mapa.set_title("Mapa de sensibilidade")
    ax_mapa.axis("off")

    fig_mapa.tight_layout()

    # RELATÓRIO NA TELA

    texto_relatorio = (
        "RADVISION IA\n"
        "Tecnologia e precisão em cada imagem.\n\n"
        "Análise computacional concluída.\n\n"
        f"Incidência informada: {incidencia}\n\n"
        "Principais achados identificados:\n"
    )

    for nome, valor in principais:
        texto_relatorio += (
            f"- {traduzir_achado(nome)}: "
            f"{float(valor):.4f}\n"
        )

    texto_relatorio += (
        "\nOs escores são experimentais e não representam "
        "probabilidades diagnósticas. O mapa de sensibilidade "
        "não confirma a localização de lesões. "
        "Esta ferramenta não substitui a avaliação "
        "de um profissional habilitado."
    )

    # PDF

    pdf = gerar_pdf(incidencia, principais)

    return (
        tabela_resultados,
        fig,
        fig_mapa,
        texto_relatorio,
        pdf
    )


# ==========================================
# CABEÇALHO
# ==========================================

coluna_logo, coluna_titulo = st.columns([1, 3])

with coluna_logo:
    if os.path.exists("IMG_5787.jpeg"):
        st.image("IMG_5787.jpeg", width=190)
    else:
        st.markdown("## 🩻")

with coluna_titulo:
    st.markdown("""
    <div style="padding-top:20px;">
        <h1>RADVISION <span style="color:#A78BFA;">IA</span></h1>
        <h3>Tecnologia e precisão em cada imagem.</h3>
        <p>Análise experimental de radiografias de tórax</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

st.markdown("## ENVIO DA RADIOGRAFIA")

coluna_imagem, coluna_resultado = st.columns(
    [1, 1.4],
    gap="large"
)

with coluna_imagem:

    arquivo = st.file_uploader(
        "Selecionar radiografia de tórax",
        type=["jpg", "jpeg", "png"]
    )

    incidencia = st.selectbox(
        "Incidência radiográfica",
        [
            "Não informada",
            "AP",
            "PA",
            "Perfil"
        ]
    )

    if arquivo is not None:
        st.image(
            arquivo,
            caption="Radiografia enviada",
            use_container_width=True
        )

    analisar = st.button(
        "ANALISAR RADIOGRAFIA",
        use_container_width=True
    )


with coluna_resultado:

    st.markdown("## RESULTADOS DA ANÁLISE")

    if analisar:

        if arquivo is None:
            st.warning("Envie uma radiografia primeiro.")

        else:

            try:
                with st.spinner(
                    "Realizando análise computacional..."
                ):

                    tabela, grafico, mapa, relatorio, pdf = (
                        analisar_radiografia(
                            arquivo,
                            incidencia
                        )
                    )

                st.success("Análise computacional concluída!")

                st.markdown("### Achados identificados")

                st.dataframe(
                    tabela,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Gráfico de resultados")

                st.pyplot(
                    grafico,
                    use_container_width=True
                )

                st.markdown("### Mapa de sensibilidade")

                st.pyplot(
                    mapa,
                    use_container_width=True
                )

                st.markdown("### Relatório experimental")

                st.text(relatorio)

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
# RODAPÉ
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <h3>RADVISION IA | Projeto acadêmico</h3>
    <p>
        Sistema experimental de análise de radiografias
        de tórax.
    </p>
    <p>
        Os resultados não substituem a avaliação
        de um profissional habilitado.
    </p>
</div>
""", unsafe_allow_html=True)

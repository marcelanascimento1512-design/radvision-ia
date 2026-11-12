
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

    tensor = torch.from_numpy(
        imagem
    ).unsqueeze(0).float()

    return imagem, tensor


# ==========================================
# CONSTRUÇÃO DO RELATÓRIO ESTRUTURADO
# ==========================================

def construir_relatorio(
    incidencia,
    escore_incidencia,
    principais
):

    linhas = []

    linhas.append(
        "RESUMO DA ANÁLISE COMPUTACIONAL"
    )

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
        linhas.append(
            "Não foram disponibilizados resultados."
        )

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

    # O primeiro modelo já foi liberado.
    # Agora carregamos o modelo de análise radiográfica.

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

    mapa = gradientes[0].max(
        dim=0
    )[0].detach().numpy()

    mapa = (mapa - mapa.min()) / (
        mapa.max() - mapa.min() + 1e-8
    )

    fig_mapa, ax_mapa = plt.subplots(figsize=(7, 7))

    ax_mapa.imshow(imagem[0], cmap="gray")
    ax_mapa.imshow(mapa, cmap="jet", alpha=0.4)

    ax_mapa.set_title("Mapa de sensibilidade")
    ax_mapa.axis("off")

    fig_mapa.tight_layout()

    # RELATÓRIO ESTRUTURADO

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
# HISTÓRICO DE ANÁLISES NA SESSÃO
# ==========================================

if "historico_analises" not in st.session_state:
    st.session_state.historico_analises = []

if "resultado_atual" not in st.session_state:
    st.session_state.resultado_atual = None


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
        type=["jpg", "jpeg", "png"],
        key="uploader_radiografia"
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

                st.session_state.resultado_atual = {
                    "incidencia": incidencia,
                    "escore_incidencia": escore_incidencia,
                    "tabela": tabela,
                    "grafico": grafico,
                    "mapa": mapa,
                    "relatorio": relatorio,
                    "pdf": pdf
                }

                registro = {
                    "data_hora": datetime.now().strftime(
                        "%d/%m/%Y às %H:%M"
                    ),
                    "arquivo": arquivo.name,
                    "incidencia": incidencia,
                    "escore_incidencia": escore_incidencia,
                    "tabela": tabela,
                    "relatorio": relatorio,
                    "pdf": pdf
                }

                st.session_state.historico_analises.insert(
                    0,
                    registro
                )

            except Exception as erro:

                st.error(
                    f"Não foi possível concluir a análise: {erro}"
                )

    resultado_atual = st.session_state.resultado_atual

    if resultado_atual is not None:

        incidencia = resultado_atual["incidencia"]
        escore_incidencia = resultado_atual["escore_incidencia"]
        tabela = resultado_atual["tabela"]
        grafico = resultado_atual["grafico"]
        mapa = resultado_atual["mapa"]
        relatorio = resultado_atual["relatorio"]
        pdf = resultado_atual["pdf"]

        st.success(
            "Análise computacional concluída!"
        )

        st.markdown("### Identificação da incidência")

        coluna_inc, coluna_escore = st.columns(2)

        with coluna_inc:
            st.metric(
                "Incidência radiográfica",
                incidencia
            )

        with coluna_escore:
            st.metric(
                "Escore do classificador",
                f"{escore_incidencia:.2f}%"
            )

        st.caption(
            "Pontuação do classificador experimental. "
            "Não representa garantia de acerto."
        )

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

        plt.close(grafico)

        st.markdown("### Mapa de sensibilidade")

        st.pyplot(
            mapa,
            use_container_width=True
        )

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


# ==========================================
# HISTÓRICO DE ANÁLISES
# ==========================================

st.divider()

st.markdown("## HISTÓRICO DE ANÁLISES")

st.caption(
    "As análises ficam disponíveis enquanto esta sessão do "
    "aplicativo estiver aberta. Ao reiniciar a sessão, o histórico será apagado."
)

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
                file_name=(
                    f"relatorio_radvision_historico_{indice + 1}.pdf"
                ),
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

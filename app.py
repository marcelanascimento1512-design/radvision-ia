
import os
import tempfile
from io import BytesIO

import streamlit as st
import numpy as np
import torch
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Configuração
st.set_page_config(
    page_title="RADVISION IA",
    page_icon="🩻",
    layout="wide"
)

torch.set_num_threads(1)

TRADUCOES = {
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

# Aparência
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0b1020;
    color: #f4f5fb;
}

h1, h2, h3 {
    color: #ffffff;
}

.stButton > button {
    background-color: #6950c9;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #8069df;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# Carregamento do modelo
@st.cache_resource
def carregar_modelo():
    modelo = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )
    modelo.eval()
    return modelo


# Gráfico
def criar_grafico(resultados):
    nomes = [item[0] for item in resultados]
    valores = [item[1] for item in resultados]

    fig, ax = plt.subplots(figsize=(9, 5))
    indices = np.argsort(valores)

    ax.barh(
        [nomes[i] for i in indices],
        [valores[i] for i in indices]
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Pontuação do modelo (0 a 1)")
    ax.set_title("Resultados da análise experimental")
    fig.tight_layout()

    return fig


# Relatório PDF
def criar_pdf(nome, incidencia, resultados):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    largura, altura = A4

    pdf.setFillColorRGB(0.10, 0.12, 0.25)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(48, altura - 55, "RADVISION IA")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        48, altura - 75,
        "Tecnologia e precisão em cada imagem."
    )

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(
        48, altura - 115,
        "Relatório de análise experimental"
    )

    pdf.setFont("Helvetica", 10)
    pdf.drawString(48, altura - 140, f"Arquivo: {nome}")
    pdf.drawString(48, altura - 158, f"Incidência informada: {incidencia}")

    y = altura - 195

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(48, y, "Achado")
    pdf.drawString(340, y, "Pontuação")

    y -= 18
    pdf.setFont("Helvetica", 10)

    for nome_achado, valor in resultados:
        if y < 70:
            pdf.showPage()
            y = altura - 60
            pdf.setFont("Helvetica", 10)

        pdf.drawString(48, y, nome_achado[:45])
        pdf.drawString(340, y, f"{valor:.3f}")
        y -= 16

    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(
        48, 42,
        "Ferramenta acadêmica experimental. Não utilizar para diagnóstico."
    )

    pdf.save()
    buffer.seek(0)

    return buffer


# Cabeçalho
col_logo, col_titulo = st.columns([1, 4])

with col_logo:
    if os.path.exists("IMG_5787.jpeg"):
        st.image("IMG_5787.jpeg", width=150)

with col_titulo:
    st.title("RADVISION IA")
    st.subheader("Tecnologia e precisão em cada imagem.")
    st.write("Análise experimental de radiografias de tórax.")

st.divider()

# Envio da radiografia
st.header("Enviar radiografia")

arquivo = st.file_uploader(
    "Selecione uma imagem de radiografia de tórax",
    type=["png", "jpg", "jpeg"]
)

incidencia = st.selectbox(
    "Selecione a incidência radiográfica",
    ["Não informada", "AP", "PA", "Perfil"]
)

if arquivo is not None:
    st.image(
        arquivo,
        caption="Radiografia enviada",
        width=600
    )

    if st.button("Analisar radiografia", use_container_width=True):

        try:
            with st.spinner("Carregando o modelo e analisando a imagem..."):

                modelo = carregar_modelo()

                with tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                ) as temporario:
                    temporario.write(arquivo.getvalue())
                    caminho = temporario.name

                try:
                    imagem = xrv.utils.load_image(caminho)

                    tensor = torch.from_numpy(
                        imagem
                    ).unsqueeze(0).float()

                    with torch.inference_mode():
                        predicao = modelo(tensor)[0].cpu().numpy()

                finally:
                    os.remove(caminho)

                patologias = list(modelo.pathologies)

                resultados = []

                for nome, valor in zip(patologias, predicao):
                    nome_pt = TRADUCOES.get(
                        nome,
                        nome.replace("_", " ")
                    )

                    resultados.append(
                        (nome_pt, float(np.clip(valor, 0, 1)))
                    )

                resultados.sort(
                    key=lambda item: item[1],
                    reverse=True
                )

                principais = resultados[:10]

            st.success("Análise concluída!")

            st.divider()
            st.header("Resultados da análise")

            st.write("**Incidência informada:**", incidencia)

            st.dataframe(
                [
                    {
                        "Achado radiográfico": nome,
                        "Pontuação": round(valor, 3)
                    }
                    for nome, valor in principais
                ],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Gráfico de resultados")

            grafico = criar_grafico(principais)
            st.pyplot(grafico)
            plt.close(grafico)

            relatorio = criar_pdf(
                arquivo.name,
                incidencia,
                principais
            )

            st.download_button(
                label="Baixar relatório em PDF",
                data=relatorio,
                file_name="relatorio_radv​​ision.pdf".replace("\u200b", ""),
                mime="application/pdf"
            )

        except Exception as erro:
            st.error(f"Não foi possível concluir a análise: {erro}")

st.divider()

st.warning(
    "RADVISION IA é uma ferramenta acadêmica experimental. "
    "As pontuações não representam probabilidades diagnósticas "
    "e não devem ser utilizadas para diagnóstico ou decisões clínicas."
)

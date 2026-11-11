
import os
import tempfile
import gc

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchxrayvision as xrv
import cv2

from transformers import AutoModel, PreTrainedModel

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# ==========================================
# IDENTIDADE VISUAL
# ==========================================

PASTA = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(PASTA, "IMG_5787.jpeg")

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root, body, .gradio-container, .gradio-container * {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #080D22, #151638) !important;
    color: white !important;
    max-width: 100% !important;
    width: 100% !important;
}

h1, h2, h3, p, label {
    color: white !important;
}

button.primary {
    background: linear-gradient(90deg, #5545D9, #8049E8) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
}

.panel, .block {
    border-radius: 15px !important;
}
"""


# ==========================================
# CARREGAMENTO DOS MODELOS
# ==========================================

print("RADVISION IA: carregando modelos...")

dispositivo = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.set_num_threads(2)

# Modelo principal de análise radiográfica
modelo = xrv.models.DenseNet(
    weights="densenet121-res224-all"
)

modelo = modelo.to(dispositivo)
modelo.eval()

print("Modelo principal carregado.")

# Compatibilidade do modelo de incidência
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

dispositivo_view = (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Modelo de identificação da incidência
modelo_view = AutoModel.from_pretrained(
    "ianpan/chest-x-ray-basic",
    trust_remote_code=True
)

modelo_view = modelo_view.eval().to(dispositivo_view)

print("Modelo de incidência carregado.")
print("RADVISION IA pronta para uso.")


# ==========================================
# TRADUÇÃO DOS ACHADOS
# ==========================================

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
# IDENTIFICAÇÃO DA INCIDÊNCIA
# ==========================================

def identificar_incidencia(caminho):

    imagem_view = cv2.imread(caminho, 0)

    if imagem_view is None:
        raise gr.Error("Não foi possível abrir a radiografia.")

    imagem_preparada = modelo_view.preprocess(imagem_view)

    entrada_view = torch.from_numpy(imagem_preparada)
    entrada_view = entrada_view.unsqueeze(0).unsqueeze(0).float()
    entrada_view = entrada_view.to(dispositivo_view)

    with torch.inference_mode():
        saida_view = modelo_view(entrada_view)

    escores_view = torch.softmax(
        saida_view["view"], dim=1
    )[0]

    incidencias = ["AP", "PA", "Perfil"]

    indice = int(torch.argmax(escores_view).item())

    incidencia = incidencias[indice]
    escore = float(escores_view[indice].item()) * 100

    return incidencia, escore


# ==========================================
# ANÁLISE DA RADIOGRAFIA
# ==========================================

def analisar_radiografia(caminho):

    if caminho is None:
        raise gr.Error("Envie uma radiografia primeiro.")

    incidencia, escore = identificar_incidencia(caminho)

    imagem = xrv.utils.load_image(caminho)
    imagem = xrv.datasets.XRayCenterCrop()(imagem)
    imagem = xrv.datasets.XRayResizer(224)(imagem)

    entrada = torch.from_numpy(imagem).unsqueeze(0).float()
    entrada = entrada.to(dispositivo)

    with torch.no_grad():
        resultado = modelo(entrada)

    pontuacoes = resultado[0].cpu().numpy()

    analise = sorted(
        zip(modelo.pathologies, pontuacoes),
        key=lambda x: x[1],
        reverse=True
    )

    principais = analise[:10]

    tabela_resultados = [
        [traduzir_achado(nome), round(float(valor), 4)]
        for nome, valor in analise
    ]

    # Gráfico
    fig, ax = plt.subplots(figsize=(14, 9))

    nomes = [traduzir_achado(x[0]) for x in principais]
    valores = [float(x[1]) for x in principais]

    ax.barh(nomes[::-1], valores[::-1])
    ax.set_title(
        "RADVISION IA | Principais resultados",
        fontsize=17,
        pad=18
    )
    ax.set_xlabel("Escore do modelo", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    # Mapa de sensibilidade
    entrada_mapa = entrada.detach().clone()
    entrada_mapa.requires_grad_(True)

    modelo.zero_grad()
    saida = modelo(entrada_mapa)

    indice_maior = int(np.argmax(pontuacoes))

    saida[0, indice_maior].backward()

    gradientes = entrada_mapa.grad.abs()

    mapa = gradientes[0].max(dim=0)[0].detach().cpu().numpy()

    mapa = (mapa - mapa.min()) / (
        mapa.max() - mapa.min() + 1e-8
    )

    fig_mapa, ax_mapa = plt.subplots(figsize=(8, 8))

    ax_mapa.imshow(imagem[0], cmap="gray")
    ax_mapa.imshow(mapa, cmap="jet", alpha=0.4)

    ax_mapa.set_title("Mapa de sensibilidade", fontsize=15)
    ax_mapa.axis("off")

    plt.tight_layout()

    # Relatório
    texto_relatorio = (
        "RADVISION IA\n"
        "Tecnologia e precisão em cada imagem.\n\n"
        "Análise computacional concluída.\n\n"
        f"Incidência identificada: {incidencia}\n"
        f"Escore de classificação: {escore:.2f}%\n\n"
        "Principais achados identificados:\n"
    )

    for nome, valor in principais:
        texto_relatorio += (
            f"- {traduzir_achado(nome)}: {float(valor):.4f}\n"
        )

    texto_relatorio += (
        "\nOs escores são experimentais e não representam "
        "probabilidades diagnósticas. O escore da incidência "
        "não garante que a classificação esteja correta. "
        "O mapa de sensibilidade não confirma a localização "
        "de lesões. Esta ferramenta não substitui a avaliação "
        "de um profissional habilitado."
    )

    # Geração do PDF
    arquivo_pdf = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    )

    caminho_pdf = arquivo_pdf.name
    arquivo_pdf.close()

    estilos = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloRADVISION",
        parent=estilos["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5545C7"),
        fontSize=22
    )

    slogan = ParagraphStyle(
        "SloganRADVISION",
        parent=estilos["Heading2"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5545C7")
    )

    elementos = [
        Paragraph("RADVISION IA", titulo),
        Paragraph(
            "Tecnologia e precisão em cada imagem.",
            slogan
        ),
        Spacer(1, 18),
        Paragraph(
            "RELATÓRIO EXPERIMENTAL",
            estilos["Heading2"]
        ),
        Paragraph(
            f"Incidência identificada: {incidencia}",
            estilos["BodyText"]
        ),
        Paragraph(
            f"Escore de classificação: {escore:.2f}%",
            estilos["BodyText"]
        ),
        Spacer(1, 12),
        Paragraph(
            "Principais achados identificados pelo modelo:",
            estilos["Heading3"]
        )
    ]

    dados_pdf = [["Achado identificado", "Escore"]]

    for nome, valor in principais:
        dados_pdf.append([
            traduzir_achado(nome),
            f"{float(valor):.4f}"
        ])

    tabela_pdf = Table(dados_pdf, colWidths=[300, 100])

    tabela_pdf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5545C7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    elementos.extend([
        Spacer(1, 10),
        tabela_pdf,
        Spacer(1, 20),
        Paragraph(
            "<b>Importante:</b> os valores apresentados são "
            "escores experimentais, não probabilidades diagnósticas. "
            "A classificação da incidência necessita de validação "
            "independente. O mapa de sensibilidade não confirma "
            "lesões. Este sistema é acadêmico e não deve ser "
            "utilizado para diagnóstico ou decisões clínicas.",
            estilos["BodyText"]
        )
    ])

    documento = SimpleDocTemplate(caminho_pdf, pagesize=A4)
    documento.build(elementos)

    return (
        "Análise computacional concluída!",
        incidencia,
        f"{escore:.2f}%",
        tabela_resultados,
        fig,
        fig_mapa,
        texto_relatorio,
        caminho_pdf
    )


# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=css,
    title="RADVISION IA"
) as demo:

    with gr.Column():

        if os.path.exists(logo_path):
            gr.Image(
                value=logo_path,
                label=None,
                interactive=False
            )

        gr.Markdown("""
        <div style="text-align:center; padding:10px;">
            <h1 style="font-size:38px; color:#FFFFFF; font-weight:800;">
                RADVISION <span style="color:#A78BFA;">IA</span>
            </h1>
            <p style="font-size:18px; color:#B8C5DD;">
                Tecnologia e precisão em cada imagem.
            </p>
            <p style="font-size:14px; color:#B8C5DD;">
                Análise experimental de radiografias de tórax
            </p>
        </div>
        """)

    gr.Markdown("---")

    with gr.Row():

        with gr.Column(scale=2):

            gr.Markdown("### ENVIO DA RADIOGRAFIA")

            imagem = gr.Image(
                type="filepath",
                label="Selecionar radiografia de tórax",
                height=600,
                interactive=True
            )

            botao = gr.Button(
                "ANALISAR RADIOGRAFIA",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ### Sobre a RADVISION IA

            Sistema acadêmico experimental para reconhecimento
            de padrões em radiografias de tórax.
            """)

        with gr.Column(scale=3):

            gr.Markdown("### RESULTADOS DA ANÁLISE")

            resultado = gr.Textbox(
                label="Situação da análise"
            )

            incidencia_detectada = gr.Textbox(
                label="Incidência identificada automaticamente"
            )

            confianca_view = gr.Textbox(
                label="Escore da classificação da incidência"
            )

            tabela = gr.Dataframe(
                headers=["Achado identificado", "Escore do modelo"],
                label="Achados identificados"
            )

            grafico = gr.Plot(
                label="Gráfico de resultados"
            )

            mapa = gr.Plot(
                label="Mapa de sensibilidade"
            )

            relatorio = gr.Textbox(
                label="Relatório experimental",
                lines=8
            )

            arquivo = gr.File(
                label="Baixar relatório em PDF"
            )

    gr.Markdown("""
    ---
    <div style="text-align:center;">
        <b>RADVISION IA | Projeto acadêmico</b>
        <p>Os resultados são experimentais e não substituem
        a avaliação de um profissional habilitado.</p>
    </div>
    """)

    botao.click(
        fn=analisar_radiografia,
        inputs=[imagem],
        outputs=[
            resultado,
            incidencia_detectada,
            confianca_view,
            tabela,
            grafico,
            mapa,
            relatorio,
            arquivo
        ]
    )


# ==========================================
# INICIALIZAÇÃO PARA O RENDER
# ==========================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)

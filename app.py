
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tempfile
import gradio as gr
import numpy as np
import torch
import torchxrayvision as xrv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

torch.set_num_threads(1)
DEVICE = torch.device("cpu")
MODEL = None

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
    "Enlarged Cardiomediastinum": "Alargamento do mediastino",
}

def obter_modelo():
    global MODEL
    if MODEL is None:
        MODEL = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        ).to(DEVICE)
        MODEL.eval()
    return MODEL

def criar_grafico(nomes, valores):
    fig, ax = plt.subplots(figsize=(9, 5))
    indices = np.argsort(valores)[::-1]
    nomes_ord = [nomes[i] for i in indices]
    vals_ord = [valores[i] for i in indices]
    ax.barh(nomes_ord[::-1], vals_ord[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pontuação do modelo (0 a 1)")
    ax.set_title("Resultados da análise experimental")
    fig.tight_layout()
    caminho = tempfile.mktemp(suffix=".png")
    fig.savefig(caminho, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return caminho

def criar_pdf(nome_arquivo, incidencia, resultados):
    caminho = tempfile.mktemp(suffix=".pdf")
    pdf = canvas.Canvas(caminho, pagesize=A4)
    largura, altura = A4
    pdf.setTitle("Relatório RADVISION IA")
    pdf.setFillColorRGB(0.10, 0.12, 0.25)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(48, altura - 55, "RADVISION IA")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(48, altura - 75, "Tecnologia e precisão em cada imagem.")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(48, altura - 115, "Relatório de análise experimental")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(48, altura - 140, f"Arquivo: {nome_arquivo}")
    pdf.drawString(48, altura - 158, f"Incidência informada: {incidencia}")
    y = altura - 195
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(48, y, "Achado")
    pdf.drawString(340, y, "Pontuação")
    y -= 18
    pdf.setFont("Helvetica", 10)
    for nome, valor in resultados:
        if y < 70:
            pdf.showPage()
            y = altura - 60
            pdf.setFont("Helvetica", 10)
        pdf.drawString(48, y, nome[:45])
        pdf.drawString(340, y, f"{valor:.3f}")
        y -= 16
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(
        48, 42,
        "Ferramenta acadêmica experimental. Não utilizar para diagnóstico ou decisão clínica."
    )
    pdf.save()
    return caminho

def analisar(imagem, incidencia):
    if imagem is None:
        raise gr.Error("Envie uma radiografia para iniciar a análise.")

    try:
        modelo = obter_modelo()
        img = xrv.utils.load_image(imagem)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE)

        with torch.inference_mode():
            pred = modelo(tensor)[0].detach().cpu().numpy()

        patologias = list(modelo.pathologies)
        resultados = []

        for nome, valor in zip(patologias, pred):
            traduzido = TRADUCOES.get(
                nome, nome.replace("_", " ")
            )
            resultados.append(
                (traduzido, float(np.clip(valor, 0, 1)))
            )

        resultados.sort(key=lambda item: item[1], reverse=True)
        principais = resultados[:10]

        tabela = [
            [nome, f"{valor:.3f}"]
            for nome, valor in principais
        ]

        grafico = criar_grafico(
            [item[0] for item in principais],
            [item[1] for item in principais]
        )

        pdf = criar_pdf(
            os.path.basename(imagem),
            incidencia,
            principais
        )

        imagem_exibicao = Image.open(imagem).convert("RGB")

        return (
            imagem_exibicao,
            incidencia,
            tabela,
            grafico,
            pdf,
            "Análise concluída. Os valores representam pontuações do modelo, não probabilidades diagnósticas."
        )

    except Exception as erro:
        raise gr.Error(
            f"Não foi possível analisar a imagem: {erro}"
        )

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .gradio-container {
    font-family: Inter, sans-serif !important;
}
.gradio-container {
    background: #0b1020 !important;
    color: #f4f5fb !important;
}
"""

logo = "IMG_5787.jpeg"

with gr.Blocks(css=CSS, title="RADVISION IA") as demo:
    with gr.Column():

        if os.path.exists(logo):
            gr.Image(
                value=logo,
                show_label=False,
                interactive=False,
                height=150,
                container=False
            )

        gr.Markdown(
            "# RADVISION IA\n"
            "### Tecnologia e precisão em cada imagem.\n"
            "**Ferramenta acadêmica experimental para estudo de radiografias de tórax.**"
        )

        gr.Markdown(
            "Envie uma imagem de radiografia de tórax. "
            "Nesta versão, informe manualmente a incidência."
        )

        arquivo = gr.Image(
            type="filepath",
            label="Radiografia de tórax",
            height=360
        )

        incidencia = gr.Radio(
            choices=["AP", "PA", "Perfil", "Não informada"],
            value="Não informada",
            label="Incidência radiográfica"
        )

        botao = gr.Button(
            "Analisar radiografia",
            variant="primary"
        )

        imagem_saida = gr.Image(
            label="Imagem enviada",
            interactive=False
        )

        incidencia_saida = gr.Textbox(
            label="Incidência informada"
        )

        tabela_saida = gr.Dataframe(
            headers=["Achado radiográfico", "Pontuação"],
            datatype=["str", "str"],
            label="Principais resultados",
            interactive=False
        )

        grafico_saida = gr.Image(
            label="Gráfico de resultados"
        )

        pdf_saida = gr.File(
            label="Relatório em PDF"
        )

        mensagem = gr.Markdown()

        botao.click(
            fn=analisar,
            inputs=[arquivo, incidencia],
            outputs=[
                imagem_saida,
                incidencia_saida,
                tabela_saida,
                grafico_saida,
                pdf_saida,
                mensagem
            ]
        )

        gr.Markdown(
            "---\n"
            "**Aviso importante:** RADVISION IA é um projeto acadêmico e experimental. "
            "Não substitui avaliação de radiologista, não confirma doenças e não deve "
            "ser usado para diagnóstico, triagem ou decisão clínica."
        )

if __name__ == "__main__":
    porta = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=porta
    )

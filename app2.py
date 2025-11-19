import streamlit as st
import json
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# --- 1. Configuração da Página ---
st.set_page_config(
    page_title="Analisador Inteligente de ECG com NeuroKit2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Funções de Análise e Dicionários de Patologia ---


def analisar_ecg(ecg_signal, sf):
    """Processa o sinal de ECG usando NeuroKit2 e extrai parâmetros."""
    if not ecg_signal or not isinstance(ecg_signal, list) or len(ecg_signal) < 100:
        return None, None, None, "Sinal de ECG inválido ou muito curto."

    # Processamento Completo
    try:
        # 1. Pré-processamento e Delineamento
        processed_signal, info = nk.ecg_process(ecg_signal, sampling_rate=sf)

        # 2. Extração de Parâmetros
        analysis = nk.ecg_analyze(processed_signal, sampling_rate=sf)

        # O NeuroKit2 retorna uma Série, converte para DataFrame para visualização
        parameters = analysis.T.rename(columns={0: "Valor"}).reset_index().rename(
            columns={'index': 'Parâmetro'})

        return processed_signal, info, parameters, None
    except Exception as e:
        return None, None, None, f"Erro durante o processamento do NeuroKit2: {e}"


def analisar_anormalidades(params_df):
    """Analisa os parâmetros extraídos e lista os fora da normalidade."""
    anormalidades = []

    # Convertermos a coluna de Parâmetros em um dicionário para fácil acesso
    params_dict = params_df.set_index('Parâmetro')['Valor'].to_dict()

    # --- Limites Médicos Padrão (Base) ---

    # 1. Frequência Cardíaca (BPM)
    hr = params_dict.get('HR_Mean', np.nan)
    if hr < 60:
        anormalidades.append(
            ("Frequência Cardíaca", f"{hr:.1f} bpm", "Bradicardia (Normal: 60-100 bpm)"))
    elif hr > 100:
        anormalidades.append(
            ("Frequência Cardíaca", f"{hr:.1f} bpm", "Taquicardia (Normal: 60-100 bpm)"))

    # 2. Intervalo PR (ms)
    pr_duration = params_dict.get(
        'ECG_P_R_Interval', np.nan) * 1000  # Convertendo s para ms
    if pr_duration < 120:
        anormalidades.append(
            ("Intervalo PR", f"{pr_duration:.1f} ms", "PR Curto (Normal: 120-200 ms)"))
    elif pr_duration > 200:
        anormalidades.append(
            ("Intervalo PR", f"{pr_duration:.1f} ms", "Bloqueio AV de 1° Grau (Normal: 120-200 ms)"))

    # 3. Duração do QRS (ms)
    # É uma aproximação no NK2, ou QRS_Duration se disponível
    qrs_duration = params_dict.get('ECG_R_S_Interval', np.nan) * 1000
    if np.isnan(qrs_duration):
        # Tentativa de achar a duração QRS
        qrs_duration = params_dict.get('ECG_QRS_Duration', np.nan) * 1000

    if qrs_duration > 120:
        anormalidades.append(
            ("Duração QRS", f"{qrs_duration:.1f} ms", "Alargado (Normal: < 120 ms)"))

    # 4. Intervalo QTc (ms) - Corrigido
    # Convertendo s para ms, usando Hodge (método padrão NK2)
    qtc_interval = params_dict.get('ECG_QTc_Hodge', np.nan) * 1000
    if qtc_interval > 440:
        anormalidades.append(
            ("Intervalo QTc", f"{qtc_interval:.1f} ms", "Longo (Normal: < 440 ms)"))

    # Adicionar mais critérios conforme a necessidade (e dados disponíveis no NK2)

    return anormalidades


def sugerir_patologias(anormalidades, achados_localizacao):
    """Gera sugestões de patologias e CID-10."""
    patologias = []

    # Patologias Baseadas em Intervalos/FC
    for param, _, condicao in anormalidades:
        if "Bradicardia" in condicao:
            patologias.append(("Bradicardia Sinusal", "R00.1"))
        elif "Taquicardia" in condicao:
            patologias.append(("Taquicardia Sinusal", "I47.1"))
        elif "Bloqueio AV de 1° Grau" in condicao:
            patologias.append(
                ("Bloqueio Atrioventricular de 1° Grau", "I44.0"))
        elif "Alargado" in condicao:
            patologias.append(
                ("Bloqueio de Ramo ou Distúrbio de Condução", "I45.9"))
        elif "Longo" in condicao:
            patologias.append(("Síndrome do QT Longo Adquirida", "I45.8"))

    # Patologias Baseadas em Localização (Simuladas)
    if achados_localizacao:
        if "ST V1-V2" in achados_localizacao or "Q V1-V2" in achados_localizacao:
            patologias.append(("IAM Septal (Agudo/Antigo)", "I21.0"))
        if "ST V3-V4" in achados_localizacao or "Q V3-V4" in achados_localizacao:
            patologias.append(("IAM Anterior (Agudo/Antigo)", "I21.0"))
        if "ST V5-aVL" in achados_localizacao or "Q V5-aVL" in achados_localizacao:
            patologias.append(("IAM Lateral (Agudo/Antigo)", "I21.2"))
        if "ST II-III-aVF" in achados_localizacao or "Q II-III-aVF" in achados_localizacao:
            patologias.append(("IAM Inferior (Agudo/Antigo)", "I21.1"))
        if "Depressão ST Multipla" in achados_localizacao:
            patologias.append(("Isquemia Subendocárdica Difusa", "I20.9"))

    # Remover duplicatas (mantendo a ordem)
    return list(dict.fromkeys(patologias))

# --- 3. Interface Streamlit ---


st.title("⚡️ Analisador Inteligente de ECG com NeuroKit2")
st.markdown("Uma ferramenta para processamento de sinais de ECG a partir de dados JSON, extração de parâmetros e sugestão de patologias.")



# 📂 Upload e Configuração

uploaded_file = st.file_uploader(
    "1. Carregue o arquivo JSON do ECG",
    type="json"
)

# Definir valores padrão para taxa de amostragem
default_sf = 1000
sf = st.number_input(
    "Taxa de Amostragem (Sampling Rate) em Hertz (Hz)",
    min_value=100,
    max_value=4000,
    value=default_sf,
    step=100
)

# Inicializar variáveis principais
ecg_data = None
ecg_signal = None
derivacao = "Não Especificada"

if uploaded_file is not None:
    try:
        # Leitura do arquivo JSON
        data = json.load(uploaded_file)

        # Supondo a estrutura de um único registro para demonstração
        if isinstance(data, list) and len(data) > 0:
            ecg_data = data[0]
        elif isinstance(data, dict):
            ecg_data = data

        if ecg_data and "ECG" in ecg_data:
            ecg_signal = ecg_data["ECG"]
            if "Derivacao" in ecg_data:
                derivacao = ecg_data["Derivacao"]
            st.success(
                f"Arquivo JSON carregado com sucesso. Derivação principal: **{derivacao}**.")

            # Atualiza SF se estiver no JSON (opcional, mas bom ter)
            if "Taxa_Amostragem" in ecg_data and st.session_state.get('sf_update', True):
                json_sf = ecg_data["Taxa_Amostragem"]
                if json_sf != sf:
                    st.warning(
                        f"O JSON sugere uma taxa de amostragem de {json_sf} Hz. Use o campo acima para ajustar, se necessário.")
                # st.session_state['sf_update'] = False # Para evitar loop de atualização

        else:
            st.error(
                "JSON inválido: Campos 'ECG' não encontrados na estrutura esperada.")
            ecg_signal = None

    except json.JSONDecodeError:
        st.error(
            "Erro ao decodificar o JSON. Certifique-se de que o arquivo está formatado corretamente.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")



# 🧠 Processamento e Análise de Sinais

if ecg_signal and st.button("2. Processar ECG com NeuroKit2"):

    with st.spinner("Analisando sinal de ECG..."):

        # Chamada da função de análise
        processed_signal, info, parameters, error_message = analisar_ecg(
            ecg_signal, sf)

    if error_message:
        st.error(f"Falha na Análise: {error_message}")
    elif parameters is not None:

        st.session_state['parameters'] = parameters
        st.session_state['processed_signal'] = processed_signal
        st.session_state['info'] = info
        st.session_state['derivacao'] = derivacao

        st.success("Processamento concluído. Parâmetros extraídos.")

        # --- 4. Visualização ---
        st.subheader("📊 Gráfico de Delineamento do ECG")

        try:
            # Gráfico do NeuroKit2 com picos marcados
            fig = nk.events_plot(info["ECG_R_Peaks"],
                                 processed_signal["ECG_Clean"])
            nk.signal_plot(processed_signal["ECG_Clean"], sampling_rate=sf,
                           subplots=True, standardize=True, show=False)

            # Adiciona os pontos de delineamento
            if "ECG_P_Onsets" in info and "ECG_T_Offsets" in info:
                # Cria a figura com Matplotlib
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(processed_signal["ECG_Clean"], label="Sinal Limpo")

                # Marcação dos picos R
                r_peaks = info["ECG_R_Peaks"]
                ax.plot(
                    r_peaks, processed_signal["ECG_Clean"][r_peaks], 'ro', label="Picos R")

                # Marcação de início/fim de ondas (opcional, se forem muitos pontos)
                # nk.events_plot(info["ECG_P_Onsets"], processed_signal["ECG_Clean"], color='green')
                # nk.events_plot(info["ECG_T_Offsets"], processed_signal["ECG_Clean"], color='purple')

                ax.set_title(
                    f"Sinal de ECG ({derivacao}) com Picos R Detectados (SF={sf} Hz)")
                ax.set_xlabel("Amostras")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
            else:
                st.info(
                    "Gráfico detalhado com P, Q, S, T não disponível. Exibindo apenas o sinal limpo.")
                fig_clean = plt.figure(figsize=(10, 4))
                plt.plot(processed_signal["ECG_Clean"])
                plt.title(f"Sinal de ECG Limpo ({derivacao})")
                st.pyplot(fig_clean)

        except Exception as e:
            st.error(f"Erro ao gerar o gráfico: {e}")

        # --- Tabela de Parâmetros ---
        st.subheader("📑 Parâmetros Detalhados do ECG (NeuroKit2)")
        st.dataframe(parameters, hide_index=True)

    else:
        st.warning(
            "Aguardando o upload do arquivo e o clique no botão 'Processar ECG'.")

# --- 5. Resultados Finais e Patologias (Após Processamento) ---

if 'parameters' in st.session_state:

    st.markdown("---")
    st.header("🚨 Análise de Anormalidades e Sugestões de Patologias")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Tabela de Parâmetros Fora da Normalidade")

        anormalidades = analisar_anormalidades(st.session_state['parameters'])

        if anormalidades:
            anormalidades_df = pd.DataFrame(
                anormalidades, columns=["Parâmetro", "Valor Encontrado", "Condição"])
            st.error("⚠️ Achados Anormais:")
            st.dataframe(anormalidades_df, hide_index=True)
        else:
            st.success(
                "✅ Todos os parâmetros básicos (FC, PR, QRS, QTc) estão dentro da faixa de normalidade esperada.")

    with col2:
        st.subheader("Simulação de Achados de Localização (Infarto/Isquemia)")

        # SIMULAÇÃO DA ANÁLISE MULTI-DERIVAÇÃO
        achados_localizacao = st.text_area(
            "Insira achados de outras derivações (Ex: ST V1-V4, Q II-III-aVF, Depressão ST Multipla)",
            value="",
            help="Este campo simula a detecção de desvios ST e ondas Q patológicas em múltiplas derivações, essencial para localizar patologias isquêmicas."
        )

        st.subheader("Lista de Potenciais Patologias e CID-10")

        patologias = sugerir_patologias(anormalidades, achados_localizacao)

        if patologias:
            patologias_df = pd.DataFrame(
                patologias, columns=["Potencial Patologia", "CID-10"])
            st.warning("Sugestões de Patologias:")
            st.dataframe(patologias_df, hide_index=True)
        else:
            st.info("Nenhuma patologia sugerida com base nos critérios atuais.")

    st.markdown("---")

    # --- Disclaimer Médico Proeminente ---
    st.markdown(
        """
        > **⚠️ DISCLAMER IMPORTANTE:**
        > Esta análise é puramente **computacional e sugerida** (baseada em critérios médicos padrão).
        > **NÃO** substitui o diagnóstico clínico de um médico cardiologista.
        > Para qualquer decisão médica, consulte um profissional de saúde qualificado.
        """
    )

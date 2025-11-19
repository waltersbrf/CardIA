import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Importa NeuroKit2 (manter import aqui)
try:
    import neurokit2 as nk
except ImportError:
    st.error("A biblioteca NeuroKit2 não está instalada. Por favor, instale-a para rodar a análise de ECG.")
    st.stop()

# --- Configuração Inicial ---
st.set_page_config(
    page_title="Analisador Inteligente de ECG Multi-Exame",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variáveis para evitar NameError no escopo global
data = None
exams_data = []
num_exams = 0
sf = 500  # Valor padrão


# --- Funções de Processamento e Análise ---

def calcular_idade(nascimento):
    """Calcula a idade a partir da data de nascimento (ISO format)."""
    try:
        data_nascimento = datetime.strptime(
            nascimento.split('T')[0], '%Y-%m-%d')
        hoje = datetime.now()
        idade = hoje.year - data_nascimento.year - (
            (hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day)
        )
        return idade
    except Exception:
        return "N/A"


def formatar_dados_paciente(data):
    """Extrai e formata os dados do paciente e histórico do JSON."""
    p = data.get('paciente', {})
    h = data.get('historico', {})
    c = h.get('comorbidade', {})
    m = h.get('medicamento', {})

    comorbidades = ", ".join(
        [k for k, v in c.items() if v is True]) if c else "Nenhuma"
    medicamentos = ", ".join(
        [k for k, v in m.items() if v is True]) if m else "Nenhum"
    if m.get('outros'):
        medicamentos += f" ({m['outros']})"

    idade = calcular_idade(p.get('nascimento', ''))

    paciente_df = pd.DataFrame({
        'Parâmetro': [
            'Nome', 'CPF', 'Data de Nasc.', 'Idade', 'Sexo',
            'Pressão Sistólica', 'Pressão Diastólica', 'Altura (m)',
            'Peso (kg)', 'Comorbidades', 'Medicamentos'
        ],
        'Valor': [
            p.get('nome', 'N/A'),
            p.get('cpf', 'N/A'),
            p.get('nascimento', 'N/A').split('T')[0],
            idade,
            p.get('sexo', 'N/A'),
            h.get('pressaoArterialSistolica', 'N/A'),
            h.get('pressaoArterialDiastolica', 'N/A'),
            h.get('altura', 'N/A'),
            h.get('peso', 'N/A'),
            comorbidades,
            medicamentos
        ]
    })
    return paciente_df


def extrair_parametros_morfologicos(signals: pd.DataFrame, sf: int) -> dict:
    """
    Extrai parâmetros morfológicos globais (médias) a partir do DataFrame de sinais do ecg_process.
    Retorna um dicionário com durações em ms e amplitudes em mV.
    """
    # Garante que temos o sinal limpo
    if "ECG_Clean" not in signals.columns:
        return {}

    ecg_clean = signals["ECG_Clean"].values

    def get_indices(colname):
        if colname in signals.columns:
            return np.where(signals[colname].values == 1)[0]
        return np.array([], dtype=int)

    # Índices dos eventos
    p_onsets = get_indices("ECG_P_Onsets")
    p_offsets = get_indices("ECG_P_Offsets")
    p_peaks = get_indices("ECG_P_Peaks")

    q_peaks = get_indices("ECG_Q_Peaks")
    s_peaks = get_indices("ECG_S_Peaks")

    r_peaks = get_indices("ECG_R_Peaks")

    t_onsets = get_indices("ECG_T_Onsets")
    t_offsets = get_indices("ECG_T_Offsets")
    t_peaks = get_indices("ECG_T_Peaks")

    def mean_interval(samples1, samples2):
        n = min(len(samples1), len(samples2))
        if n == 0:
            return np.nan
        # intervalo em segundos
        return float(np.mean((samples2[:n] - samples1[:n]) / sf))

    def mean_amp(idxs):
        if len(idxs) == 0:
            return np.nan
        return float(np.mean(ecg_clean[idxs]))

    features = {}

    # Durações em milissegundos (ms)
    features["P Dur. (ms)"] = mean_interval(p_onsets, p_offsets) * 1000
    features["PR (ms)"] = mean_interval(p_onsets, r_peaks) * 1000
    features["QRS Dur. (ms)"] = mean_interval(q_peaks, s_peaks) * 1000
    features["QT (ms)"] = mean_interval(q_peaks, t_offsets) * 1000
    features["ST (ms)"] = mean_interval(s_peaks, t_onsets) * 1000

    # Amplitudes
    features["P Ampl. (mV)"] = mean_amp(p_peaks)
    features["R Ampl. (mV)"] = mean_amp(r_peaks)
    features["S Ampl. (mV)"] = mean_amp(s_peaks)
    features["T Ampl. (mV)"] = mean_amp(t_peaks)

    return features


def processar_derivacao(ecg_signal, sf, derivacao_name):
    """Processa um sinal individual, extrai parâmetros e trata a saída."""

    # Conversão de unidades (assumindo ganho de 10 do JSON de exemplo ⇒ dividindo por 100)
    if ecg_signal and isinstance(ecg_signal, list):
        ecg_signal = [s / 100.0 for s in ecg_signal]

    if not ecg_signal or len(ecg_signal) < 100:
        empty_data = {'Status': [f"Sinal Curto/Inválido: {derivacao_name}"]}
        return pd.DataFrame(empty_data, index=[derivacao_name]), None, f"Sinal inválido para {derivacao_name}"

    try:
        # Processa ECG
        processed_signal, info = nk.ecg_process(ecg_signal, sampling_rate=sf)
        # Análise automática (FC média + HRV)
        analysis = nk.ecg_analyze(processed_signal, sampling_rate=sf)

        # --- 1) Extrair recursos escalares da análise (sem listas/arrays) ---
        features = {}
        row0 = analysis.iloc[0]

        for k, v in row0.items():
            # Ignorar valores não escalares (listas, arrays, dicts...)
            if isinstance(v, (list, np.ndarray, dict, pd.Series)):
                continue

            # Renomear FC para rótulo amigável
            if k in ["HR_Mean", "ECG_Rate_Mean"]:
                key = "FC (BPM)"
            else:
                key = k

            # Se houver colisão de nomes, o último valor prevalece
            features[key] = float(v) if isinstance(
                v, (np.floating, np.integer)) else v

        # --- 2) Adicionar parâmetros morfológicos globais (em ms e mV) ---
        morfo = extrair_parametros_morfologicos(processed_signal, sf)
        features.update(morfo)

        # --- 3) Montar DataFrame final por derivação ---
        if not features:
            empty_data = {'Status': [
                f"Nenhum parâmetro escalar extraído: {derivacao_name}"]}
            return pd.DataFrame(empty_data, index=[derivacao_name]), processed_signal, None

        params_df = pd.DataFrame([features])
        params_df.index = [derivacao_name]
        params_df.index.name = 'Derivacao'

        return params_df, processed_signal, None

    except Exception as e:
        empty_data = {'Status': [f"Erro: {e}"]}
        return pd.DataFrame(empty_data, index=[derivacao_name]), None, f"Erro NeuroKit2: {e}"


def compilar_estatisticas_por_derivacao(full_raw_df_list):
    """Compila e calcula a média, mediana e moda de todos os parâmetros para cada derivação em todos os blocos."""

    if not full_raw_df_list:
        return pd.DataFrame()

    full_df = pd.concat(full_raw_df_list)

    compiled_stats = []

    param_columns = [
        col for col in full_df.columns if col not in ['Exame', 'Derivacao']]

    for derivacao in full_df['Derivacao'].unique():
        df_deriv = full_df[full_df['Derivacao'] == derivacao].copy()
        stats_row = {'Derivacao': derivacao}

        for param in param_columns:
            if param in df_deriv.columns:
                values = df_deriv[param].apply(
                    pd.to_numeric, errors='coerce').dropna()

                if not values.empty:
                    stats_row[f'{param} - Média'] = values.mean()
                    stats_row[f'{param} - Mediana'] = values.median()
                    stats_row[f'{param} - Moda'] = (
                        values.mode().iloc[0] if not values.mode(
                        ).empty else np.nan
                    )

        if len(stats_row) > 1:
            compiled_stats.append(stats_row)

    stats_df = pd.DataFrame(compiled_stats)

    # Arredondar colunas numéricas
    for col in stats_df.columns:
        if col not in ['Derivacao'] and pd.api.types.is_numeric_dtype(stats_df[col]):
            stats_df[col] = stats_df[col].round(4)

    if stats_df.empty:
        return stats_df

    cols = ['Derivacao'] + \
        [col for col in stats_df.columns if col != 'Derivacao']
    return stats_df[cols]


# --- Função para gerar Laudo Automático (texto único + conclusão) ---

def gerar_laudo_ecg(final_stats_df: pd.DataFrame, data_json: dict) -> str:
    """
    Gera um laudo textual único, sintetizando os principais achados
    e, ao final, apresentando uma CONCLUSÃO com possível correlação clínica.
    Estilo técnico, porém compreensível.
    """

    if final_stats_df.empty:
        return (
            "Não foi possível gerar o laudo automático, pois nenhum parâmetro "
            "estatístico numérico foi identificado nas derivações analisadas."
        )

    # --- Dados do paciente / contexto clínico ---
    p = data_json.get("paciente", {}) if data_json else {}
    h = data_json.get("historico", {}) if data_json else {}
    c = h.get("comorbidade", {}) if isinstance(h, dict) else {}
    m = h.get("medicamento", {}) if isinstance(h, dict) else {}

    idade = calcular_idade(p.get("nascimento", "")) if p else "N/A"
    sexo = p.get("sexo", "N/A") if p else "N/A"
    press_sis = h.get("pressaoArterialSistolica", None) if h else None
    press_dias = h.get("pressaoArterialDiastolica", None) if h else None

    comorbidades_list = [k for k, v in c.items(
    ) if v is True] if isinstance(c, dict) else []
    medicamentos_list = [k for k, v in m.items() if isinstance(
        v, bool) and v] if isinstance(m, dict) else []
    outros_meds = m.get("outros") if isinstance(m, dict) else None

    # --- Parâmetros globais do ECG (média entre derivações) ---
    col_fc_media = "FC (BPM) - Média"
    col_pr = "PR (ms) - Média"
    col_qrs = "QRS Dur. (ms) - Média"
    col_qt = "QT (ms) - Média"
    col_st = "ST (ms) - Média"
    col_t = "T Ampl. (mV) - Média"

    def media_global(col):
        if col in final_stats_df.columns:
            val = final_stats_df[col].mean()
            return float(val) if pd.notna(val) else None
        return None

    fc_global = media_global(col_fc_media)
    pr_global = media_global(col_pr)
    qrs_global = media_global(col_qrs)
    qt_global = media_global(col_qt)
    st_global = media_global(col_st)
    t_global = media_global(col_t)

    # --- Montagem do texto descritivo único ---

    partes_texto = []

    # 1) Introdução (paciente + exames)
    intro_paciente = "Trata-se de exame de eletrocardiograma processado de forma automatizada."
    if idade != "N/A" or sexo != "N/A":
        sexo_txt = ""
        if isinstance(sexo, str) and sexo.upper().startswith("M"):
            sexo_txt = "do sexo masculino"
        elif isinstance(sexo, str) and sexo.upper().startswith("F"):
            sexo_txt = "do sexo feminino"

        if idade != "N/A" and isinstance(idade, int):
            if sexo_txt:
                intro_paciente = f"Trata-se de exame de eletrocardiograma de paciente {sexo_txt}, com aproximadamente {idade} anos."
            else:
                intro_paciente = f"Trata-se de exame de eletrocardiograma de paciente com aproximadamente {idade} anos."
        else:
            if sexo_txt:
                intro_paciente = f"Trata-se de exame de eletrocardiograma de paciente {sexo_txt}."
    partes_texto.append(intro_paciente)

    if comorbidades_list:
        partes_texto.append(
            "Há registro de comorbidades referidas, entre elas: "
            + ", ".join(comorbidades_list) + "."
        )

    if medicamentos_list or outros_meds:
        meds_descr = []
        if medicamentos_list:
            meds_descr.append(", ".join(medicamentos_list))
        if outros_meds:
            meds_descr.append(f"outros: {outros_meds}")
        partes_texto.append(
            "Consta uso de medicações: " + "; ".join(meds_descr) + "."
        )

    if press_sis is not None and press_dias is not None:
        partes_texto.append(
            f"Os registros de pressão arterial no momento do exame são em torno de {press_sis}/{press_dias} mmHg."
        )

    # 2) Descrição dos achados eletrocardiográficos (global)
    descricoes_ecg = []

    # Frequência cardíaca
    if fc_global is not None:
        if fc_global < 60:
            descricoes_ecg.append(
                f"A frequência cardíaca média estimada é em torno de {fc_global:.0f} bpm, em faixa de bradicardia."
            )
        elif fc_global > 100:
            descricoes_ecg.append(
                f"A frequência cardíaca média estimada é em torno de {fc_global:.0f} bpm, em faixa de taquicardia."
            )
        else:
            descricoes_ecg.append(
                f"A frequência cardíaca média estimada é de aproximadamente {fc_global:.0f} bpm, dentro da faixa usual de normalidade."
            )

    # Intervalo PR
    if pr_global is not None:
        if pr_global > 200:
            descricoes_ecg.append(
                f"O intervalo PR global apresenta-se prolongado (cerca de {pr_global:.0f} ms), achado compatível com bloqueio atrioventricular de primeiro grau."
            )
        elif pr_global < 120:
            descricoes_ecg.append(
                f"O intervalo PR global encontra-se encurtado (cerca de {pr_global:.0f} ms), podendo sugerir a presença de via acessória ou pré-excitação, dependendo da morfologia do traçado."
            )
        else:
            descricoes_ecg.append(
                f"O intervalo PR médio situa-se em torno de {pr_global:.0f} ms, dentro dos limites considerados usuais."
            )

    # QRS
    if qrs_global is not None:
        if qrs_global > 120:
            descricoes_ecg.append(
                f"A duração média do complexo QRS é alargada (aproximadamente {qrs_global:.0f} ms), sugerindo distúrbio de condução intraventricular, como bloqueio de ramo, a ser melhor caracterizado no traçado completo."
            )
        else:
            descricoes_ecg.append(
                f"A duração média do complexo QRS é de cerca de {qrs_global:.0f} ms, sem evidência de alargamento significativo."
            )

    # QT
    if qt_global is not None:
        descricoes_ecg.append(
            f"O intervalo QT médio encontra-se em torno de {qt_global:.0f} ms. A interpretação mais precisa deste achado exige correção pela frequência cardíaca (QTc) "
            "e correlação com idade, sexo e uso de fármacos."
        )

    # ST (tempo, não amplitude)
    if st_global is not None:
        descricoes_ecg.append(
            f"A duração média do segmento ST é estimada em aproximadamente {st_global:.0f} ms. Este valor se refere ao tempo do segmento, não permitindo, isoladamente, concluir sobre elevação ou depressão do ST."
        )

    # Onda T
    if t_global is not None:
        if t_global < 0:
            descricoes_ecg.append(
                f"A amplitude média das ondas T apresenta tendência à negatividade (cerca de {t_global:.2f} mV), o que pode refletir alteração de repolarização ventricular e deve ser correlacionado com sintomas e demais exames."
            )
        else:
            descricoes_ecg.append(
                f"A amplitude média das ondas T é positiva (em torno de {t_global:.2f} mV), sem inversão global importante nas derivações analisadas."
            )

    if descricoes_ecg:
        partes_texto.append(
            "Do ponto de vista eletrocardiográfico global, observam-se os seguintes aspectos principais: "
            + " ".join(descricoes_ecg)
        )

    # Se nada relevante foi descrito:
    if not descricoes_ecg:
        partes_texto.append(
            "Nos parâmetros globais avaliados (frequência cardíaca, intervalos e amplitudes), não se destacam alterações significativas, "
            "mantendo-se o traçado dentro de padrões considerados usuais na média das derivações."
        )

    # --- Construção da CONCLUSÃO (potencial patologia) ---

    conclusoes_automatizadas = []

    # FC
    if fc_global is not None:
        if fc_global < 60:
            conclusoes_automatizadas.append(
                "bradicardia sinusal ou ritmo lento.")
        elif fc_global > 100:
            conclusoes_automatizadas.append(
                "taquicardia sinusal ou ritmo acelerado.")

    # PR
    if pr_global is not None and pr_global > 200:
        conclusoes_automatizadas.append(
            "bloqueio atrioventricular de primeiro grau.")

    # QRS
    if qrs_global is not None and qrs_global > 120:
        conclusoes_automatizadas.append(
            "distúrbio de condução intraventricular (por exemplo, bloqueio de ramo).")

    # T global
    if t_global is not None and t_global < 0:
        conclusoes_automatizadas.append(
            "alteração inespecífica da repolarização ventricular.")

    # Comorbidades ajudam a colorir a conclusão
    texto_comorb = ""
    if comorbidades_list:
        texto_comorb = (
            " Considerando a presença de comorbidades descritas ("
            + ", ".join(comorbidades_list)
            + "), tais achados podem estar relacionados à condição clínica de base e ao uso de medicações."
        )

    # Montagem da seção Conclusão
    texto_conclusao = ""
    if conclusoes_automatizadas:
        # Junta as possíveis patologias em texto corrido
        if len(conclusoes_automatizadas) == 1:
            pat = conclusoes_automatizadas[0]
            texto_conclusao = (
                f"Os achados automatizados sugerem, em primeiro plano, padrão compatível com {pat}"
                + texto_comorb
            )
        else:
            # Lista sem ficar "engessado"
            principais = "; ".join(conclusoes_automatizadas)
            texto_conclusao = (
                f"O conjunto dos parâmetros analisados é compatível com, entre outras possibilidades, "
                f"{principais}" + texto_comorb
            )
    else:
        texto_conclusao = (
            "À luz dos parâmetros globais avaliados, não há evidência automatizada de alteração eletrocardiográfica "
            "que sugira uma patologia específica de forma isolada. O traçado, na média dos índices calculados, "
            "mostra-se compatível com padrão de normalidade ou variações fisiológicas."
        )

    # --- Montagem final do laudo em Markdown ---

    laudo_markdown = []

    laudo_markdown.append("## LAUDO AUTOMATIZADO DE ELETROCARDIOGRAMA\n")
    laudo_markdown.append(" ".join(partes_texto) + "\n")

    laudo_markdown.append("## 🔎 Conclusão\n")
    laudo_markdown.append(
        texto_conclusao
        + " Ressalta-se que esta interpretação é gerada por algoritmo e não substitui "
          "a análise detalhada do traçado por médico(a) cardiologista ou profissional habilitado."
    )

    return "\n\n".join(laudo_markdown)


# --- Interface da Aplicação ---
st.markdown("## Analisador de ECG Multi-Exame e Estatísticas")

# --- 1. Upload e Exibição de Dados do Paciente ---
uploaded_file = st.file_uploader(
    "1. Carregue o arquivo JSON contendo os exames de ECG",
    type="json"
)

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
        exams_data = data.get('conteudosExame', [])
        num_exams = len(exams_data)

        if num_exams == 0:
            st.error(
                "Estrutura JSON inválida: A chave 'conteudosExame' está vazia ou ausente.")
            st.stop()

        st.success(
            f"Arquivo JSON carregado com sucesso. Encontrados **{num_exams} exames (blocos)** para análise.")

        paciente_df = formatar_dados_paciente(data)
        st.dataframe(paciente_df, hide_index=True, width=700)

        sf_default = exams_data[0].get('taxaAmostragem', 500)
        sf = st.number_input(
            "Taxa de Amostragem (Sampling Rate) em Hertz (Hz)",
            min_value=100,
            max_value=4000,
            value=sf_default,
            step=100,
            key='sf_input'
        )

    except json.JSONDecodeError:
        st.error(
            "Erro ao decodificar o JSON. Certifique-se de que o arquivo está formatado corretamente.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro durante a leitura inicial do JSON: {e}")
        st.stop()

    st.markdown("---")

    # --- 2. Processamento e Exibição de Resultados ---
    if st.button(f"2. Iniciar Análise Detalhada dos {num_exams} Exames"):

        all_blocks_results = []
        full_raw_df_list = []

        with st.spinner(f"Processando {num_exams} blocos de exame..."):

            for i, bloco in enumerate(exams_data):
                block_num = i + 1

                derivations_dfs = []
                current_signals = {}

                derivacoes = bloco.get('derivacoes', [])
                current_sf = bloco.get('taxaAmostragem', sf)

                for derivacao_data in derivacoes:

                    derivacao_name = derivacao_data.get('descricao', 'N/A')
                    ecg_signal = derivacao_data.get('amostra', [])

                    params_df, processed_signal, error = processar_derivacao(
                        ecg_signal, current_sf, derivacao_name
                    )

                    if 'Status' not in params_df.columns:
                        derivations_dfs.append(params_df)
                        # Armazenar sinal normalizado (já em mV) para gráfico
                        current_signals[derivacao_name] = np.array(
                            ecg_signal) / 100.0
                    elif error:
                        st.warning(
                            f"Bloco {block_num} / Derivação {derivacao_name}: {error}")

                if derivations_dfs:
                    consolidated_df = pd.concat(derivations_dfs)
                    consolidated_df['Exame'] = block_num

                    all_blocks_results.append({
                        'block_num': block_num,
                        'consolidated_df': consolidated_df,
                        'signals': current_signals,
                        'sf': current_sf
                    })

                    # Prepara o DataFrame para a compilação final,
                    # transformando o índice 'Derivacao' em coluna
                    df_compilacao = consolidated_df.copy(
                    ).reset_index(names=['Derivacao'])
                    full_raw_df_list.append(df_compilacao)

        # --- 3. Exibição Detalhada por Bloco e Gráficos ---

        if all_blocks_results:
            st.header("🔬 Resultados Detalhados do Processamento de ECG")

            for result in all_blocks_results:

                st.markdown(
                    f"### 📑 Tabela de Parâmetros - Exame #{result['block_num']}")

                pivot_df = result['consolidated_df'].drop(columns=['Exame'])

                # Segurança extra: remover colunas com valores não escalares (se ainda existir algo estranho)
                def is_non_scalar(x):
                    return isinstance(x, (list, np.ndarray, dict, pd.Series))

                non_scalar_cols_pivot = [
                    col for col in pivot_df.columns
                    if pivot_df[col].apply(is_non_scalar).any()
                ]
                if non_scalar_cols_pivot:
                    pivot_df = pivot_df.drop(
                        columns=non_scalar_cols_pivot, errors='ignore')

                st.dataframe(pivot_df.round(4), width=1200)

                # --- Gráficos: DII em destaque + grid 4xN ---

                st.subheader(
                    f"📈 Gráficos por Derivação - Exame #{result['block_num']}")

                signals = result["signals"]

                # 1️⃣ DII sozinho, mais comprido
                if "DII" in signals:
                    st.markdown("### 🫀 Derivação DII (principal)")
                    fig, ax = plt.subplots(figsize=(12, 3.5))
                    ax.plot(signals["DII"], linewidth=0.8)
                    ax.set_title("DII", fontsize=14)
                    ax.set_xlabel("Amostras", fontsize=10)
                    ax.set_ylabel("mV", fontsize=10)
                    ax.tick_params(labelsize=8)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(
                        "⚠️ A derivação DII não foi encontrada neste exame.")

                # 2️⃣ Grid 4×N com todas as derivações (incluindo DII de novo)
                st.markdown(
                    "### 📊 Outras Derivações (incluindo DII novamente)")

                # Ordem preferida, se as derivações existirem
                preferred_order = ["V1", "V2", "V3", "V4",
                                   "V5", "V6", "DI", "DII",
                                   "DIII", "aVR", "aVL", "aVF"]

                deriv_names = [d for d in preferred_order if d in signals] + \
                    [d for d in signals.keys() if d not in preferred_order]

                cols_per_row = 4

                for i in range(0, len(deriv_names), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for col_idx, deriv_idx in enumerate(range(i, min(i + cols_per_row, len(deriv_names)))):
                        deriv_name = deriv_names[deriv_idx]
                        signal = signals[deriv_name]

                        with cols[col_idx]:
                            st.markdown(f"**{deriv_name}**")
                            fig, ax = plt.subplots(figsize=(4, 2))
                            ax.plot(signal, linewidth=0.8)
                            ax.set_title(deriv_name, fontsize=10)
                            ax.set_xlabel("Amostras", fontsize=8)
                            ax.set_ylabel("mV", fontsize=8)
                            ax.tick_params(labelsize=7)
                            st.pyplot(fig)
                            plt.close(fig)

            # --- Tabela Final Compilada de Estatísticas ---

            if full_raw_df_list:
                final_stats_df = compilar_estatisticas_por_derivacao(
                    full_raw_df_list)

                # Segurança extra: garantir que não haja colunas não escalares
                if not final_stats_df.empty:
                    def is_non_scalar_fs(x):
                        return isinstance(x, (list, np.ndarray, dict, pd.Series))

                    non_scalar_cols_fs = [
                        col for col in final_stats_df.columns
                        if final_stats_df[col].apply(is_non_scalar_fs).any()
                    ]
                    if non_scalar_cols_fs:
                        final_stats_df = final_stats_df.drop(
                            columns=non_scalar_cols_fs, errors='ignore')

                st.header("📊 Compilação Estatística Global por Derivação")
                st.markdown(
                    "Média, Mediana e Moda de todos os parâmetros por derivação, "
                    "considerando **todos os exames carregados**."
                )

                if not final_stats_df.empty:
                    st.dataframe(final_stats_df, width=1200)

                    # --- Laudo automático baseado nas estatísticas globais + contexto do paciente ---
                    st.header("📝 Laudo Automático Gerado")
                    laudo_texto = gerar_laudo_ecg(final_stats_df, data)
                    st.markdown(laudo_texto)
                else:
                    st.info(
                        "Nenhuma estatística numérica pôde ser calculada com os dados disponíveis.")

            st.markdown(
                """
> **⚠️ AVISO IMPORTANTE:**
> Esta análise é puramente **computacional e sugerida** (baseada em parâmetros eletrocardiográficos).
> **NÃO** substitui o diagnóstico de um(a) médico(a) cardiologista ou profissional habilitado.
                """
            )
        else:
            st.error("Nenhum exame válido foi processado.")

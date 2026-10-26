# DeepchecksIA – Bike Sharing: Monitoramento de Concept Drift com Deepchecks

Projeto de Ciência de Dados e Machine Learning focado na construção, validação e **monitoramento contínuo** de um modelo de regressão com o dataset **Bike Sharing** (Capital Bikeshare, Washington D.C., 2011–2012), usando a biblioteca **[Deepchecks](https://docs.deepchecks.com/)**.

O dataset foi escolhido porque tem **drift real**: o sistema cresceu muito de 2011 para 2012 enquanto o clima e o calendário ficaram parecidos. É um caso de *concept drift*, o tipo mais traiçoeiro para monitoramento: as entradas não mudam, mas o modelo passa a errar.

---

## 📋 Sumário

- [Visão Geral e Objetivos](#-visão-geral-e-objetivos)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Dataset](#-dataset)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [Janelas Temporais](#-janelas-temporais)
- [Relatórios e Validação (Deepchecks)](#-relatórios-e-validação-deepchecks)
- [Resultados](#-resultados)
- [Evidently × Deepchecks](#-evidently--deepchecks)
- [Como Executar o Projeto](#-como-executar-o-projeto)
- [Decisões Técnicas](#-decisões-técnicas)
- [Limitações Conhecidas](#-limitações-conhecidas)

---

## 🎯 Visão Geral e Objetivos

1. **Modelagem:** prever aluguéis por hora (`cnt`) a partir de calendário e clima, com target em `log1p` e treino **somente em 2011**.
2. **Integridade dos Dados:** verificar a qualidade do lote de produção (2012).
3. **Drift:** medir mudanças nas features (*Feature Drift*), no target (*Label Drift*), nas previsões (*Prediction Drift*) e no conjunto (*Multivariate Drift*).
4. **Gate Automatizado (PASS/FAIL):** suíte customizada com *conditions* que aprova ou reprova o lote.
5. **Monitoramento mensal:** rodar o Deepchecks mês a mês em 2012, comparando com o mesmo mês de 2011.
6. **Resposta ao drift:** comparar re-treino ingênuo, janela recente e recalibração, e validar a escolha no gate.

---

## 📂 Estrutura do Repositório

```text
deepchecks_bikesharing/
├── README.md
├── requirements.txt                   # Versões fixadas p/ compatibilidade do Deepchecks 0.19
├── .gitignore
├── data/
│   ├── raw/
│   │   └── hour.csv                   # Dataset bruto (UCI #275), 17.379 linhas
│   └── processed/
│       └── bike_clean.parquet         # Dataset limpo
├── models/
│   ├── pipeline_2011.joblib           # Pipeline treinada (ColumnTransformer + LightGBM), alvo em log1p
│   └── metrics.json                   # Janelas, métricas de validação, decisão sobre correção de viés
├── notebooks/
│   ├── data_cleaning.ipynb            # 1: Limpeza, decodificação, clima em unidades reais, horas ausentes
│   ├── eda.ipynb                      # 2: EDA e diagnóstico do drift (features × target)
│   ├── modelling.ipynb                # 3: Treino em 2011, baseline, teste da correção de viés
│   └── deepchecks.ipynb               # 4: Suítes, gate, monitoramento mensal e estratégias
├── src/
│   ├── __init__.py
│   └── bike.py                        # Features, pipeline, CountModel e ScaledModel (compartilhados)
└── runs/
    ├── deepchecks_data_integrity.html         # Suíte Data Integrity (2012)
    ├── deepchecks_train_test_validation.html  # Suíte Train-Test Validation (2011 × 2012)
    ├── deepchecks_model_evaluation.html       # Suíte Model Evaluation
    ├── deepchecks_gate.html                   # Gate de produção (PASS / FAIL)
    ├── deepchecks_gate_summary.csv            # Gate em tabela
    ├── deepchecks_monthly_2012.csv / .png     # Monitoramento mês a mês
    ├── deepchecks_strategies_2012H2.png       # Comparação das estratégias de resposta
    └── deepchecks_recalibration_2012H2.csv    # Gate mensal: modelo atual × recalibrado
```

---

## 📊 Dataset

**Bike Sharing Dataset** — UCI Machine Learning Repository #275 (Fanaee-T & Gama, 2013), licença CC BY 4.0.

| Item | Descrição / Valor |
|---|---|
| **Arquivo bruto** | `data/raw/hour.csv` (17.379 linhas, 17 colunas, 1 linha por hora) |
| **Período** | 01/01/2011 a 31/12/2012 |
| **Target** | `cnt` — aluguéis na hora (= `casual` + `registered`) |
| **Features** | `hour`, `month`, `season`, `weekday`, `holiday`, `workingday`, `weather`, `temp`, `atemp`, `humidity`, `windspeed` |
| **Fora do modelo** | `casual`, `registered` (vazamento: somam `cnt`), `year`, `date`, `datetime` |

### Por que serve para testar monitoramento

| | 2011 | 2012 |
|---|---|---|
| Média de aluguéis/hora | 143,8 | 234,7 |
| Total de aluguéis | 1,24 milhão | 2,05 milhões |

- O uso de 2012 é **1,4× a 2,5×** o do mesmo mês de 2011, e os usuários cadastrados crescem 68%;
- O drift das features é baixo (KS ≤ 0,07 no ano), mas o do target é alto (KS = 0,22).

---

## ⚙️ Pipeline de Machine Learning

```text
data/raw/  ──>  data_cleaning.ipynb  ──>  data/processed/  ──>  eda.ipynb
                                                 │
                                                 ▼
                                          modelling.ipynb  ──>  models/
                                                 │
                                                 ▼
                                          deepchecks.ipynb ──>  runs/
```

* **`data_cleaning.ipynb`:** renomeia colunas, cria `datetime`, decodifica `season` (o README da UCI está errado: 1 = inverno), `weather` e `weekday`, desfaz a normalização do clima (°C e % de umidade; vento na escala original da UCI), marca `humidity = 0` como ausente (falha de sensor em 10/03/2011) e documenta as 165 horas que faltam no arquivo (incluindo o furacão Sandy, 29–30/10/2012).
* **`eda.ipynb`:** sazonalidade, perfil por hora (pico casa–trabalho em dias úteis), efeito do clima e a comparação 2011 × 2012 coluna a coluna, que mostra o *concept drift*.
* **`modelling.ipynb`:**
  * Validação em blocos dentro de 2011 (dias ≥ 24 de cada mês), sem misturar horas vizinhas entre treino e validação;
  * Baseline (média por hora × dia útil): RMSE 83,3. Modelo LightGBM em `log1p(cnt)`: **RMSE 41,1, R² 0,91**;
  * **Teste da correção de viés log-normal:** aqui ela **piora** o resultado (RMSE 44,0; soma prevista +8%) porque os resíduos em log têm variância bem diferente entre a madrugada e o resto do dia. O notebook decide pela validação e não aplica a correção;
  * O modelo final é o treinado nos dias 1–23; os dias 24+ de 2011 (que ele nunca viu) viram a **janela de referência** do monitoramento.
* **`deepchecks.ipynb`:** suítes, gate, monitoramento mensal e estratégias (detalhes abaixo).

---

## 🕒 Janelas Temporais

| Janela | Período | Papel |
|---|---|---|
| **Treino** | 2011, dias 1–23 de cada mês | treinar o modelo |
| **Referência** | 2011, dias 24+ de cada mês (holdout) | `train_dataset` nas suítes |
| **Produção** | 2012 inteiro | `test_dataset` nas suítes |
| **Mensal** | cada mês de 2012 × o mesmo mês de 2011 | monitoramento contínuo (evita alarmes de sazonalidade) |

> Por que a referência é um holdout e não o treino: no próprio treino o RMSE é 19; no holdout é 41. Com o treino como referência, qualquer lote novo pareceria degradado.

---

## 📈 Relatórios e Validação (Deepchecks)

| Arquivo | Suíte | O que verifica |
|---|---|---|
| `deepchecks_data_integrity.html` | `data_integrity()` em 2012 | nulos, duplicados, tipos mistos, correlação entre features, correlação data × target |
| `deepchecks_train_test_validation.html` | `train_test_validation()` | *Feature/Label/Multivariate Drift*, categorias novas, vazamento por data |
| `deepchecks_model_evaluation.html` | `model_evaluation()` | desempenho, distribuição do erro, segmentos fracos |
| `deepchecks_gate.html` | `Suite` customizada | regras de produção (abaixo) |

### Regras do gate

| Checagem | Regra |
|---|---|
| `FeatureDrift` | no máximo 2 features com drift ≥ 0.2 |
| `MultivariateDrift` | drift global < 0.75 |
| `NewCategoryTrainTest` | ≤ 1% de linhas com categoria nova |
| `PredictionDrift` | drift das previsões < 0.15 |
| `LabelDrift` | drift do `cnt` < 0.15 |
| `TrainTestPerformance` | degradação relativa < 30% |
| `RegressionErrorDistribution` | erro médio < 30% do RMSE (viés) |
| `PercentOfNulls` / `DataDuplicates` | ≤ 5% nulos por coluna; ≤ 1% duplicados |

---

## 🧾 Resultados

### Gate 2012 — ❌ REPROVADO

| Checagem | Status | Valor |
|---|---|---|
| Label Drift | ❌ | 0.21 |
| Train Test Performance | ❌ | RMSE 41 → 130, MAE +255%, R² 0.91 → 0.61 |
| Erro sistemático (2012) | ❌ | 68% do RMSE; soma prevista −38% |
| Feature Drift | ✅ | nenhuma feature ≥ 0.2 |
| Multivariate Drift | ✅ | 0.30 (AUC 0.65) |
| **Prediction Drift** | ✅ | **0.03** |
| Categorias novas, nulos, duplicados | ✅ | — |

**A lição central:** as previsões não mudaram (Prediction Drift = 0.03) porque as entradas não mudaram. Um monitor baseado só em entradas e previsões diria que está tudo bem com um modelo que subestima a demanda em 38%. Só as checagens que usam o **rótulo real** pegam o problema.

### Outros achados

- **`Identifier Label Correlation`:** só o `datetime` prevê parte do `cnt` em 2012 (PPS 0.36), um sinal do crescimento que aparece sem nem comparar com 2011.
- **`Weak Segments Performance`:** o pior segmento em 2012 é **quarta a sexta, 17h–19h** (RMSE ≈ 249 contra 130 de média), o pico de volta do trabalho dos usuários cadastrados.
- **Monitoramento mensal:** desempenho e erro sistemático reprovam nos 12 meses (viés de −57% em janeiro a −27% em maio); Label Drift reprova em 11; Prediction Drift passa em todos. O Feature Drift capta clima real (jan/2012 +3 °C, mar/2012 +5 °C, nov/2012 −3 °C em relação a 2011).
- **Furacão Sandy:** outubro/2012 tem 36 horas a menos e nenhuma checagem do Deepchecks percebe (horas ausentes não viram nulos). Um monitor de volume cobriria isso.

### Estratégias no 2º semestre de 2012

| Estratégia | RMSE médio mensal | Viés médio |
|---|---|---|
| A — modelo atual | 129,5 | −34% |
| B — re-treino ingênuo (2011 + 1º sem. 2012) | 131,4 | −34% |
| C — só 1º semestre de 2012 | 79,5 | −11% |
| **D — recalibração (fator das últimas 4 semanas)** | **62,0** | **+0,2%** |

- **Re-treinar não resolveu:** nenhuma feature representa o tamanho da base de usuários, então mais dados não ajudam.
- **A recalibração zera o viés:** com ela, o erro sistemático passa em todos os meses, mas o SLA de RMSE (≤ 1,5× validação = 62) passa só em 3 de 6. É mitigação, não solução.
- **Armadilha de validação:** a regra de degradação *relativa* aprovaria a recalibração por engano, porque o fator também distorce as previsões da referência. Para comparar estratégias, o notebook usa um limite absoluto.

---

## 🔁 Evidently × Deepchecks

| Evidently | Deepchecks |
|---|---|
| `Report` + `DataDriftPreset` | `train_test_validation()` → `FeatureDrift`, `MultivariateDrift` |
| `TargetDriftPreset` | `LabelDrift` (rótulo real) e `PredictionDrift` (saída do modelo) |
| `RegressionPreset` | `model_evaluation()` → `TrainTestPerformance`, `RegressionErrorDistribution`, `WeakSegmentsPerformance` |
| — | `data_integrity()` |
| `TestSuite` + `Test*` | `Suite` + `check.add_condition_*()` |
| recebe colunas `target` / `prediction` prontas | recebe o **modelo** (por isso o `CountModel` em `src/bike.py`) |

---

## 🚀 Como Executar o Projeto

> Testado com **Python 3.11**. As versões do `requirements.txt` são fixadas porque o Deepchecks 0.19 tem problemas com scikit-learn e plotly mais novos.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Execute os notebooks em ordem (pasta `notebooks/`):

1. `data_cleaning.ipynb`
2. `eda.ipynb`
3. `modelling.ipynb`
4. `deepchecks.ipynb` (~1 minuto)

Abra os HTMLs de `runs/` em qualquer navegador. Eles são autocontidos (~5 MB cada) e funcionam offline.

---

## 📌 Decisões Técnicas

| Decisão | Motivo |
|---|---|
| Treino só em 2011 | 2012 fica inteiro como produção, sem vazamento |
| Validação em blocos de dias | horas vizinhas são quase idênticas; *split* aleatório superestimaria o modelo |
| Referência = holdout de 2011 | comparação justa entre referência e produção |
| `log1p(cnt)` **sem** correção de viés | a validação mostrou que a correção piora (resíduos heterocedásticos) |
| `casual` / `registered` fora das features | `cnt = casual + registered` |
| `humidity = 0` → NaN | umidade de 0% é falha de sensor (um dia inteiro, 10/03/2011) |
| Horas ausentes não preenchidas | não se sabe se seriam 0 aluguéis ou sistema fora do ar |
| `cat_features` e `datetime_name` explícitos no `Dataset` | evita inferência errada de tipos e habilita checagens temporais |
| Monitoramento mensal contra o mesmo mês de 2011 | evita alarmes de drift causados por sazonalidade |
| Código compartilhado em `src/bike.py` | o mesmo pipeline/wrapper no treino, no Deepchecks e no re-treino |

---

## ⚠️ Limitações Conhecidas

- **Holdout mensal pequeno:** cada mês de referência tem ~170 horas, então o RMSE de referência mensal oscila (59 em abril, 75 em novembro).
- **Sem features de demanda recente:** o modelo não tem *lags* (mesma hora ontem / semana passada), e é por isso que nem o re-treino ingênuo nem a recalibração resolvem tudo. É o próximo passo natural.
- **Compatibilidade:** o Deepchecks 0.19 exige scikit-learn < 1.4 para `WeakSegmentsPerformance` funcionar em regressão; `Boosting Overfit` não reconhece o wrapper e aparece como N/A.

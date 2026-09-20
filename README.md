# Previsão de Popularidade de Músicas — Spotify

Projeto de ciência de dados que explora quais fatores mais influenciam a popularidade de uma faixa no Spotify, combinando análise exploratória com modelos preditivos.

## Pergunta de negócio

O que torna uma música popular: características de áudio (dançabilidade, energia, acústica...) ou o gênero musical em si? E até que ponto é possível prever a popularidade de uma faixa usando apenas dados técnicos disponíveis publicamente?

## Dataset

- **114.000 faixas**, 20 colunas (dados técnicos de áudio + metadados)
- Fonte: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle)
- Colunas incluem: `danceability`, `energy`, `loudness`, `acousticness`, `speechiness`, `valence`, `tempo`, `track_genre`, entre outras, além da variável-alvo `popularity` (0–100)

## Limpeza dos dados

- Removidas **24.259 duplicatas** de `track_id`
- Identificadas e tratadas duplicatas por combinação de `artists` + `track_name`
- Removidas linhas com valores nulos em `artists`, `album_name` e `track_name`
- Removidos registros com valores zerados suspeitos em `tempo`, `duration_ms` e `time_signature`

## Metodologia

Testamos três modelos para prever a popularidade das faixas:

### 1. KNN
Modelo baseado em distância entre vizinhos mais próximos, com as features numéricas normalizadas via `StandardScaler` — essencial nesse tipo de modelo, já que features em escalas muito diferentes (ex: `loudness` em dB negativos vs. `duration_ms` na casa dos milhões) distorcem o cálculo de proximidade.

### 2. Random Forest
Modelo baseado em ensemble de árvores de decisão, capaz de capturar interações não-lineares entre as features sem exigir normalização.

### 3. XGBoost
Modelo de gradient boosting, testado como alternativa mais rápida ao Random Forest.

| Modelo | MAE | R² |
|---|---|---|
| KNN (com StandardScaler) | ~10.9 | ~0.36 |
| XGBoost | ~13.2 | ~0.34 |
| **Random Forest** | **~10.5** | **~0.51** |

*(valores exatos no notebook — sujeitos a pequenas variações entre execuções)*

**Principais achados:**
- Random Forest teve o melhor desempenho geral, capturando interações não-lineares entre as features.
- XGBoost treina significativamente mais rápido que Random Forest, um trade-off relevante para cenários de produção.
- **Gênero musical, quando analisado de forma agregada (soma da importância de todas as categorias), é o fator individual mais determinante da popularidade** — mais do que qualquer feature de áudio isolada, mesmo aparecendo fragmentado entre 114 colunas no one-hot encoding.

## Como rodar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook spotify.ipynb
```

O dataset (`dataset.csv`) já está incluído neste repositório.

## Limitações

Nenhum modelo atingiu um R² alto, o que sugere que grande parte da popularidade de uma faixa depende de fatores fora deste dataset — marketing, presença do artista nas redes sociais, inclusão em playlists editoriais, entre outros.

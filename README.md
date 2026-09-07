# Progetto di Fondamenti e Applicazioni del Machine Learning — A.A. 2026

Dataset assegnato: **Dataset 1 — MOOC User Action Dataset** (*act-mooc*, Stanford SNAP).
Task: classificazione binaria, prevedere l'**abbandono** dello studente.

## Stato

| task | descrizione | stato |
|---|---|---|
| 1 | Preprocessing e preparazione dati | ✅ completato |
| 2 | Classificatori costruiti a mano su `manuale.csv` | ✅ completato |
| 3 | Data quality ed EDA su `training.csv` | ✅ completato |
| 4 | Valutazione dei classificatori manuali sui dati completi | ✅ completato |
| 5 | Modelli Scikit-Learn e scelta del classificatore finale | ✅ completato |

## Struttura

```
Project/
├── data/
│   ├── manuale.csv              12 studenti (6 abbandoni + 6 no), per il Task 2
│   ├── training.csv             7.035 studenti, per i Task 3-5
│   └── act-mooc/                i tre TSV originali (non versionati)
├── notebooks/
│   ├── 01_preprocessing.ipynb   Task 1: dal livello azione al livello studente
│   ├── 02.1_naive_bayes.ipynb   Task 2: Naive Bayes costruito a mano
│   ├── 02.2_albero_decisione.ipynb  Task 2: albero di decisione costruito a mano
│   ├── 03_analisi_esplorativa.ipynb Task 3: data quality ed EDA
│   ├── 04_valutazione.ipynb     Task 4: ottimizzazione dei classificatori manuali
│   ├── 05_modellazione.ipynb    Task 5: modelli sklearn e classificatore finale
│   └── preprocessing.py         la trasformazione del Task 1, per il file d'esame
├── documentation/
│   ├── 00_sintesi.md            guida allo studio: filo conduttore e domande d'orale
│   ├── 01_preprocessing.md … 05_modellazione.md
│   ├── figure/                  figure estratte dai notebook
│   └── costruisci_pdf.sh        genera il PDF unico (fuori dal progetto)
├── README.md
└── requirements.txt
```

## Come riprodurre

I dati grezzi non sono versionati. Vanno scaricati da
<https://snap.stanford.edu/data/act-mooc.html> ed estratti in `data/`, così da ottenere
`data/act-mooc/mooc_actions.tsv`, `mooc_action_features.tsv` e `mooc_action_labels.tsv`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/  # eseguire i notebook in ordine, dall'alto in basso
```

Il notebook rigenera `manuale.csv` e `training.csv`. Tutte le estrazioni casuali usano
`random_state=42`, quindi il risultato è identico a ogni esecuzione.

## Il file d'esame

Le 8 feature di `training.csv` sono definite da noi e non esistono nel dataset originale: il
`real_settings.csv` fornito in sede d'esame arriverà quindi come log di azioni, da trasformare con
la stessa funzione usata in addestramento.

```python
import preprocessing as pp
X, y = pp.carica_per_predire("real_settings.csv")   # log di azioni o tabella gia' aggregata
previsioni = finale.predict(X)                      # `finale` viene da 05_modellazione.ipynb
```

Il classificatore finale è una **regressione logistica** (`StandardScaler`, `C = 0,1`): 79,25% di
accuratezza sul test set. Non è salvato su disco — si riottiene eseguendo `05_modellazione.ipynb`,
che impiega meno di un minuto.

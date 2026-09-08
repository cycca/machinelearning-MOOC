# Previsione dell'abbandono in una piattaforma MOOC

Progetto di Fondamenti e Applicazioni del Machine Learning, A.A. 2026.
Studenti: **Federico Ciccarelli**, **Lorenzo Lallone**. Docenti: prof. Fabrizio Rossi, prof. Fabio
Persia.

Dataset *act-mooc* (Stanford SNAP): 411.749 azioni di 7.047 studenti. Classificazione binaria —
prevedere l'abbandono.

Il classificatore finale è una regressione logistica: **79,25%** di accuratezza sul test set, contro
una baseline del 57,71%.

## Struttura

```
data/            manuale.csv (12 studenti), training.csv (7.035), act-mooc/ (grezzi, non versionati)
notebooks/       i sei notebook, in ordine, + preprocessing.py e predici.py
documentation/   relazione.md, un .md di motivazioni per task, le figure
```

`documentation/relazione.md` è la relazione del progetto. Accanto, un documento per task con le
motivazioni estese e le alternative scartate, da `01_preprocessing.md` a `05_modellazione.md` — il
Task 2 è diviso in due perché il gruppo è di due componenti. `00_2kno.md` sono appunti privati di
preparazione all'orale e non fa parte della consegna.

## Far girare il progetto

I dati grezzi non sono versionati (53 MB). Scaricali da
<https://snap.stanford.edu/data/act-mooc.html> ed estraili in `data/`, così da ottenere
`data/act-mooc/mooc_actions.tsv` e gli altri due TSV.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Esegui i notebook in ordine, dall'alto in basso. Il primo rigenera `manuale.csv` e `training.csv`:
tutte le estrazioni casuali usano `random_state=42`, quindi i file escono identici a ogni
esecuzione.

**Attenzione.** I notebook leggono i dati con percorsi relativi alla propria cartella
(`../data/...`). Se li lanci da altrove, quei percorsi non si risolvono. `preprocessing.py` invece
risolve il percorso rispetto al file del modulo, quindi funziona da qualunque directory.

## Come è scritto il codice

**Nessun ciclo Python sui dati.** Ogni calcolo che tocca righe, colonne o soglie candidate è
un'operazione su array o su `DataFrame`: il Naive Bayes somma i logaritmi con un prodotto
matriciale, la ricerca dello split valuta tutte le coppie (feature, soglia) in un'unica matrice,
l'albero predice scendendo a maschere invece che riga per riga. I `for` rimasti scorrono modelli,
pieghe di cross-validation, iperparametri o assi di un grafico — dove ogni giro fa un'operazione
diversa e non c'è niente da vettorizzare.

Il Task 4 passa così da **67,7 a 6,6 secondi**, a parità di risultati: le due versioni sono state
confrontate riga per riga e i notebook 04 e 05 producono output identici al carattere. Motivazione
estesa in `documentation/relazione.md`, sezione «Come è scritto il codice».

## Usare il modello sul file d'esame

Le 8 feature sono definite da noi e non esistono nel dataset originale, quindi `real_settings.csv`
va trasformato prima di darlo al modello.

Copia il file in `data/real_settings.csv` e scegli una delle due strade.

**Dal notebook.** Apri `05_modellazione.ipynb` ed esegui tutte le celle: l'ultima cella di codice
della sezione 8 rileva il file, lo trasforma, predice e salva `data/previsioni.csv`. Se il file non
c'è, quella cella stampa un avviso e non fa altro.

**Da terminale**, senza aprire niente:

```bash
cd notebooks
python predici.py ../data/real_settings.csv
```

Entrambe le strade passano per `preprocessing.carica_per_predire`, che accetta log di azioni o
tabella già a livello studente, con o senza etichette: se le etichette ci sono stampa anche
l'accuratezza, altrimenti solo le previsioni. Il modello non è salvato su disco — si riaddestra in
un secondo, e sullo stesso seme dà previsioni identiche nei due percorsi.

## Limiti dichiarati

- La relazione «poca attività → abbandono» è in parte **tautologica**: un abbandono è per
  definizione la fine dell'attività. Su una finestra iniziale di osservazione l'AUC scende da 0,877
  a **0,535**.
- Le feature di quantità sono correlate fra loro fino a **0,91**: i coefficienti del modello non
  sono interpretabili singolarmente.
- Tre feature su otto sono quasi prive di segnale, per un difetto del dataset originale
  (`FEATURE3` ha lo stesso valore nel 93,5% delle azioni).
- `n_azioni` e `durata_giorni` dipendono dall'ampiezza della finestra di osservazione (29,77
  giorni): un file con una finestra diversa produrrebbe feature su scala diversa.

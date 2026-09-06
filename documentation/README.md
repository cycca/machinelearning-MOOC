# Documentazione del progetto

**Dataset:** *act-mooc* (Stanford SNAP) — il registro delle azioni di studenti su una piattaforma
MOOC. 411.749 azioni, 7.047 studenti, 97 attività, circa 30 giorni. `LABEL = 1` marca l'azione dopo
la quale lo studente abbandona il corso.

**Task:** classificazione binaria — prevedere l'abbandono dello studente.

Qui stanno le motivazioni delle scelte. I notebook contengono il codice e i numeri che lo
verificano: le spiegazioni per esteso sono in questi file.

## Indice

| documento | task | contenuto |
|---|---|---|
| [01_preprocessing.md](01_preprocessing.md) | 1 | dal livello azione al livello studente, le 8 feature, `manuale.csv` e `training.csv` |

Da produrre: `02_classificatori_manuali.md` (Task 2), `03_analisi_esplorativa.md` (Task 3),
`04_valutazione.md` (Task 4), `05_modellazione.md` (Task 5).

## Schema di ogni sezione

```
## N. Titolo
**Cosa abbiamo fatto.**       le operazioni, in due righe
**Perché.**                   la motivazione
**Alternative considerate.**  cosa è stato scartato e per quale ragione
**Cosa abbiamo verificato.**  i controlli stampati dal notebook
```

## File del progetto

| file | ruolo |
|---|---|
| `01_preprocessing.ipynb` | esecuzione del Task 1 e verifiche numeriche |
| `preprocessing.py` | la trasformazione del Task 1 come funzione, per il file d'esame |
| `manuale.csv` | 12 campioni (6+6) per i classificatori manuali del Task 2 |
| `training.csv` | 7.035 campioni per i Task 3, 4 e 5 |
| `data/act-mooc/` | i tre TSV originali (non versionati, vedi `.gitignore`) |

## Quattro regole valide per tutti i task

1. **`USERID` non è mai una feature.** È solo un identificativo e resta come indice per
   tracciabilità. I file vanno letti con `pd.read_csv(..., index_col="USERID")` e le feature sono
   `X = df.drop(columns="ABBANDONO")`.
2. **Lo scaling si stima dentro una `Pipeline`, sulle sole pieghe di addestramento.** Mai sui file
   salvati, mai sul file d'esame `real_settings.csv`.
3. **Le classi sono bilanciate** (57,7% di abbandoni): nessun riequilibrio, nessun SMOTE, nessun
   peso di classe. Lo sbilanciamento estremo esiste solo a livello azione, cioè nell'unità di
   analisi sbagliata.
4. **Il file d'esame passa da `preprocessing.costruisci_studenti`.** Le 8 feature sono definite da
   noi e non esistono nel dataset originale, quindi `real_settings.csv` arriverà come log di
   azioni e va trasformato con la stessa funzione usata in addestramento. Vedi
   [01_preprocessing.md](01_preprocessing.md#9-preprocessingpy-e-il-file-desame).

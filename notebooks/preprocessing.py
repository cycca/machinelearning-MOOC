"""
Preprocessing del dataset act-mooc — dal livello azione al livello studente.

Questo modulo impacchetta la trasformazione descritta nel Task 1
(`01_preprocessing.ipynb`, `documentation/01_preprocessing.md`) in una funzione
richiamabile, perché serve due volte:

1. in addestramento, per costruire `manuale.csv` e `training.csv`;
2. in sede d'esame, sul file `real_settings.csv` fornito dal docente.

Il secondo punto è la ragione per cui il modulo esiste. Le otto feature di
`training.csv` sono definite da noi e non compaiono nel dataset originale: il
docente non può quindi consegnare un file che le contenga già, e `real_settings.csv`
arriverà con ogni probabilità nel formato nativo di act-mooc, cioè un log di azioni.
`costruisci_studenti` accetta entrambi i formati e restituisce sempre la stessa
tabella, così il percorso di preprocessing usato all'esame è identico a quello
usato in addestramento.
"""

from pathlib import Path

import pandas as pd

# I dati grezzi stanno in Project/data/act-mooc/, questo modulo in Project/notebooks/.
# Il percorso è risolto rispetto al file del modulo e non alla cartella di lavoro,
# così `carica_azioni()` funziona da qualunque directory venga lanciata.
CARTELLA_DATI = Path(__file__).resolve().parent.parent / "data" / "act-mooc"

COLONNE_FEATURE = ["FEATURE0", "FEATURE1", "FEATURE2", "FEATURE3"]
COLONNE_STUDENTE = ["n_azioni", "n_attivita_distinte", "n_giorni_attivi", "durata_giorni",
                    "feature0_media", "feature1_media", "feature2_media", "feature3_media"]
SECONDI_IN_UN_GIORNO = 86400


def carica_azioni(cartella=None):
    """Unisce i tre TSV di act-mooc in un'unica tabella a livello azione.

    L'unione è **per posizione** e non con un `merge`: in `mooc_action_labels.tsv`
    la colonna ACTIONID non è una chiave (396.633 valori distinti su 411.749 righe),
    e un merge perderebbe 15.116 righe duplicandone altrettante, restituendo lo
    stesso numero di righe di partenza e passando quindi inosservato. I tre file
    hanno lo stesso numero di righe nello stesso ordine, quindi la riga i-esima
    descrive la stessa azione in tutti e tre.
    """
    cartella = Path(cartella) if cartella is not None else CARTELLA_DATI
    azioni = pd.read_csv(cartella / "mooc_actions.tsv", sep="\t")
    feature = pd.read_csv(cartella / "mooc_action_features.tsv", sep="\t")
    etichette = pd.read_csv(cartella / "mooc_action_labels.tsv", sep="\t")

    if not (len(azioni) == len(feature) == len(etichette)):
        raise ValueError("i tre file non hanno lo stesso numero di righe: "
                         f"{len(azioni)}, {len(feature)}, {len(etichette)}")

    azioni[COLONNE_FEATURE] = feature[COLONNE_FEATURE].to_numpy()
    azioni["LABEL"] = etichette["LABEL"].to_numpy()
    return azioni


def costruisci_studenti(df, verbose=False):
    """Da righe-azione a righe-studente, con le otto feature del Task 1.

    Accetta due formati:

    - una tabella **già a livello studente** (riconosciuta dalla presenza di
      `n_azioni`): viene restituita invariata, con USERID come indice se presente;
    - un **log di azioni** con almeno USERID, TARGETID, TIMESTAMP e FEATURE0..3:
      viene trasformato. La colonna LABEL è facoltativa — se c'è, si aggiunge il
      target ABBANDONO = max(LABEL) per studente; se manca, si restituiscono le
      sole feature, come serve per predire su un file non etichettato.

    L'ultima azione di **ogni** studente viene esclusa dal calcolo delle feature,
    perché è la riga su cui l'etichetta è definita. La regola si applica a tutti e
    non solo a chi abbandona, altrimenti dipenderebbe dal target.
    """
    if "n_azioni" in df.columns:                      # è già a livello studente
        return df.set_index("USERID") if "USERID" in df.columns else df

    mancanti = {"USERID", "TARGETID", "TIMESTAMP", *COLONNE_FEATURE} - set(df.columns)
    if mancanti:
        raise ValueError(f"colonne assenti dal log di azioni: {sorted(mancanti)}")

    # ordine cronologico dentro ogni studente; ACTIONID rompe gli eventuali pari merito
    ordine = ["USERID", "TIMESTAMP"] + (["ACTIONID"] if "ACTIONID" in df.columns else [])
    dati = df.sort_values(ordine, kind="stable")

    storia = dati.drop(index=dati.groupby("USERID").tail(1).index)
    persi = dati["USERID"].nunique() - storia["USERID"].nunique()
    if persi:
        print(f"attenzione: {persi} studenti con una sola azione restano senza feature "
              "e non compaiono nel risultato")

    gruppi = storia.assign(GIORNO=storia["TIMESTAMP"] // SECONDI_IN_UN_GIORNO).groupby("USERID")
    studenti = pd.DataFrame({
        "n_azioni": gruppi.size(),
        "n_attivita_distinte": gruppi["TARGETID"].nunique(),
        "n_giorni_attivi": gruppi["GIORNO"].nunique(),
        "durata_giorni": (gruppi["TIMESTAMP"].max() - gruppi["TIMESTAMP"].min()) / SECONDI_IN_UN_GIORNO,
        **{c.lower() + "_media": gruppi[c].mean() for c in COLONNE_FEATURE},
    })

    if "LABEL" in dati.columns:                       # file etichettato: aggiungiamo il target
        studenti["ABBANDONO"] = dati.groupby("USERID")["LABEL"].max()

    studenti = studenti.round(3)
    if verbose:
        print(f"{len(dati)} azioni -> {len(studenti)} studenti "
              f"({len(storia)} azioni usate per le feature)")
    return studenti


def carica_per_predire(percorso, cartella_grezzi=None):
    """Prepara un file d'esame (`real_settings.csv`) per il classificatore finale.

    Restituisce `(X, y)`, dove `y` è `None` se il file non è etichettato. Qualunque
    sia il formato del file — log di azioni o tabella già a livello studente — X ha
    sempre le otto colonne di `training.csv`, nello stesso ordine.
    """
    df = pd.read_csv(percorso, sep=None, engine="python")
    studenti = costruisci_studenti(df)
    y = studenti["ABBANDONO"] if "ABBANDONO" in studenti.columns else None
    return studenti[COLONNE_STUDENTE], y

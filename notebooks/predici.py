"""

Ricostruisce il classificatore finale del Task 5 — regressione logistica con
`C = 0,1` e `StandardScaler`, addestrata sullo stesso 80% di `training.csv` usato
nel notebook — e lo applica al file passato. I parametri non sono ricercati qui:
sono quelli che la `GridSearchCV` della sezione 4 di `05_modellazione.ipynb`
sceglie, fissati perche' lo script parta in un secondo. Il modello non e' salvato
su disco: riaddestrarlo evita i problemi di compatibilita' fra versioni di
Scikit-Learn, e sullo stesso seme da' predizioni identiche al notebook.

Il file d'esame puo' essere un log di azioni o una tabella gia' a livello
studente, etichettato o no: se ne occupa `preprocessing.carica_per_predire`.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessing as pp

# Come in preprocessing.py, i percorsi si risolvono rispetto al file del modulo:
# lo script funziona da qualunque cartella venga lanciato.
TRAINING = Path(__file__).resolve().parent.parent / "data" / "training.csv"
SEME = 42


def addestra_finale():
    """Il modello scelto nel Task 5, addestrato sullo stesso training set."""
    training = pd.read_csv(TRAINING).set_index("USERID")
    y, X = training["ABBANDONO"], training.drop(columns="ABBANDONO")
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEME)
    modello = Pipeline([("scala", StandardScaler()),
                        ("modello", LogisticRegression(C=.1, max_iter=2000, random_state=SEME))])
    return modello.fit(X_tr, y_tr)


def main(argomenti):
    if len(argomenti) != 1:
        print(__doc__.strip())
        return 2

    percorso = Path(argomenti[0])
    if not percorso.exists():
        print(f"file non trovato: {percorso.resolve()}")
        return 1

    finale = addestra_finale()
    X_esame, y_esame = pp.carica_per_predire(percorso)   # stampa un avviso se le storie sono troncate
    previsioni = finale.predict(X_esame)
    probabilita = finale.predict_proba(X_esame)[:, 1]

    print(f"\n{percorso.name}: {len(X_esame)} studenti, "
          f"{previsioni.sum()} previsti in abbandono ({previsioni.mean():.1%})")

    if y_esame is not None:
        print(f"accuratezza: {accuracy_score(y_esame, previsioni):.4f} "
              f"(riferimento: 0,7925 sul test set del Task 5)\n")
        print(classification_report(y_esame, previsioni,
                                    target_names=["prosegue", "abbandona"], digits=4))
    else:
        print("il file non ha etichette: nessuna accuratezza da calcolare\n")

    uscita = percorso.with_name("previsioni.csv")
    pd.DataFrame({"previsione": previsioni, "probabilita_abbandono": probabilita.round(4)},
                 index=X_esame.index).to_csv(uscita)
    print(f"previsioni salvate in {uscita}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

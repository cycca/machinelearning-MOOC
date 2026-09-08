# Sintesi del progetto — guida allo studio

**Corso.** Fondamenti e Applicazioni del Machine Learning, A.A. 2026 — prof. Fabrizio Rossi,
prof. Fabio Persia.
**Studenti.** Federico Ciccarelli, Lorenzo Lallone.
**Dataset.** *act-mooc* (Stanford SNAP): 411.749 azioni di 7.047 studenti su 97 attività di una
piattaforma MOOC, in circa 30 giorni.
**Task.** Classificazione binaria — prevedere l'**abbandono** dello studente.
**Risultato finale.** Regressione logistica (`StandardScaler`, `C = 0,1`): **79,25%** di accuratezza
sul test set, F1 0,8272, ROC-AUC 0,8485. Baseline: 57,71%.

Questo documento serve a **studiare il progetto**, non a sostituire i cinque documenti di dettaglio:
ogni sezione rimanda al file corrispondente. La sezione 10 raccoglie le domande più probabili
all'orale con la risposta.

---

## 1. Mappa dei file

```
Project/
├── data/            manuale.csv (12 studenti), training.csv (7.035), act-mooc/ (TSV grezzi)
├── notebooks/       i sei notebook + preprocessing.py
├── documentation/   i sette .md, le figure e lo script che genera il PDF
└── requirements.txt
```

| notebook | documento | contenuto |
|---|---|---|
| `01_preprocessing.ipynb` | `01_preprocessing.md` | dai tre TSV grezzi a `manuale.csv` e `training.csv` |
| `02.1_naive_bayes.ipynb` | `02.1_naive_bayes.md` | primo classificatore manuale (Naive Bayes) |
| `02.2_decision_tree.ipynb` | `02.2_decision_tree.md` | secondo classificatore manuale (decision tree ID3) |
| `03_analisi_esplorativa.ipynb` | `03_analisi_esplorativa.md` | data quality ed EDA |
| `04_valutazione.ipynb` | `04_valutazione.md` | ottimizzazione dei classificatori manuali |
| `05_modellazione.ipynb` | `05_modellazione.md` | modelli Scikit-Learn e scelta finale |

`notebooks/preprocessing.py` non ha un documento proprio: è la trasformazione del Task 1
impacchettata come funzione, e serve in sede d'esame su `real_settings.csv`.

**Tre regole valide ovunque**, da citare se interrogati sul metodo:

1. `USERID` non è mai una feature: è un identificativo e resta come indice.
2. Lo scaling si stima **dentro una `Pipeline`**, sulle sole pieghe di addestramento.
3. Le classi sono bilanciate (57,7%): nessun SMOTE, nessun peso di classe.

---

## 2. Il filo conduttore

Il progetto racconta **una sola storia**, e saperla esporre in trenta secondi vale più di ricordare
i numeri:

> Il dataset non è tabellare, è una sequenza di eventi. Il lavoro vero è stato **cambiare l'unità di
> analisi** — dall'azione allo studente — perché l'etichetta descrive lo studente, non l'azione.
> Quel passaggio ha dissolto uno sbilanciamento apparente (0,99% → 57,7%) e ha reso il problema
> trattabile. Da lì in poi, ogni modello — nostro o preso da Scikit-Learn, semplice o
> complesso — si ferma **intorno al 79%**, perché il limite non è nel modello ma in quanta
> informazione le feature contengono.

Tutto il resto sono conferme di questa frase, ottenute per strade indipendenti.

---

## 3. Task 1 — Preprocessing

*Dettaglio completo in* `01_preprocessing.md`

**Il problema.** Tre file TSV allineati per riga: azioni, feature, etichette. `LABEL = 1` marca
l'**ultima azione** di chi abbandona: è quindi una proprietà **dello studente**, non dell'azione.

**Tre decisioni da saper difendere.**

**a) Unione per posizione, non `merge`.** `ACTIONID` non è una chiave nel file delle etichette
(396.633 valori distinti su 411.749 righe). Un `merge` perde 15.116 righe e ne duplica altre 15.116,
restituendo **esattamente lo stesso numero di righe**: l'errore non si vedrebbe controllando `len()`.

```python
dati = azioni.copy()
dati[COLONNE_FEATURE] = feature[COLONNE_FEATURE].to_numpy()   # affianca per posizione
dati["LABEL"] = etichette["LABEL"].to_numpy()
```

**b) Cambio di unità di analisi.** Da 411.749 azioni a 7.047 studenti. La prevalenza passa da
**0,99% a 57,7%**: lo sbilanciamento estremo esisteva solo nell'unità di analisi sbagliata.

**c) Esclusione dell'ultima azione di *ogni* studente**, non solo dei positivi. Se la togliessimo
solo a chi abbandona, la regola di costruzione dipenderebbe dal target: sarebbe leakage.

```python
storia = dati.drop(index=dati.groupby("USERID").tail(1).index)
```

**Le otto feature**, tutte calcolate sulla storia così ridotta:

```python
gruppi = storia.assign(GIORNO=storia["TIMESTAMP"] // 86400).groupby("USERID")
studenti = pd.concat([
    pd.DataFrame({
        "n_azioni":            gruppi.size(),
        "n_attivita_distinte": gruppi["TARGETID"].nunique(),
        "n_giorni_attivi":     gruppi["GIORNO"].nunique(),
        "durata_giorni":       (gruppi["TIMESTAMP"].max() - gruppi["TIMESTAMP"].min()) / 86400,
    }),
    gruppi[COLONNE_FEATURE].mean().rename(columns=lambda c: c.lower() + "_media"),
], axis=1)
```

Le 404.702 righe vengono scorse una volta sola, dentro pandas. Anche le quattro medie escono da
**una** aggregazione su tutte e quattro le colonne insieme, non da quattro giri di ciclo.

**I due file prodotti.** `manuale.csv` (12 studenti, 6 + 6, estratti a caso con `random_state=42`) e
`training.csv` (i restanti 7.035).

---

## 4. Task 2 — I due classificatori manuali

*Dettaglio completo in* `02.1_naive_bayes.md`, `02.2_decision_tree.md`

Gruppo di due componenti → due classificatori. Scelti perché **complementari** e perché sono gli
unici del programma i cui calcoli si possono svolgere per intero su dodici righe.

| | decision tree | Naive Bayes |
|---|---|---|
| tipo | discriminativo | generativo |
| feature | una alla volta | tutte insieme |
| output | regole esplicite | probabilità |
| ipotesi forte | separabilità a soglie | indipendenza condizionale |

### 4.1 Decision tree (ID3, information gain)

Feature continue → soglie sui **punti medi** fra valori consecutivi. Non discretizziamo: così è
l'information gain a scegliere il confine, invece di sceglierlo noi.

```python
def entropia_binaria(positivi, totali):        # scritta sui CONTEGGI, non sulle etichette
    p = np.divide(positivi, totali, out=np.zeros_like(totali, float), where=totali > 0)
    q = 1 - p
    return -(np.where(p > 0, p * np.log2(p), 0) + np.where(q > 0, q * np.log2(q), 0)) + 0.0

def guadagno(a_sinistra, y):                   # a_sinistra: (campioni x split candidati)
    positivi, n = (y.to_numpy() == 1)[:, None], len(y)
    n_sx, pos_sx = a_sinistra.sum(axis=0), (a_sinistra & positivi).sum(axis=0)
    return (entropia(y) - (n_sx / n)       * entropia_binaria(pos_sx, n_sx)
                        - ((n - n_sx) / n) * entropia_binaria(positivi.sum() - pos_sx, n - n_sx))
```

**Perché è scritta così.** `entropia_binaria` prende i *conteggi* invece del vettore di etichette,
e diventa un'espressione di soli operatori aritmetici: NumPy la applica elemento per elemento,
quindi la stessa riga vale per un nodo o per trecento split in parallelo. `a_sinistra` ha una riga
per campione e una colonna per split candidato, così tutte le coppie (feature, soglia) del nodo si
valutano in una passata. Vedi la sezione 9.

**Risultato.** `n_attivita_distinte <= 22,5` ottiene **IG = 1,0000**, il massimo: separa i 12
campioni senza errori. L'albero è **un solo nodo**. Accuratezza 1,0000 su `manuale.csv`, 0,7734 su
`training.csv`.

**Controprova.** Togliendo la feature dominante, l'albero cresce su due livelli e resta a 12/12 —
ma alla radice **cinque feature su sette pareggiano** con lo stesso IG: quella che finisce
nell'albero è solo la prima nell'ordine delle colonne.

### 4.2 Naive Bayes (discretizzato, Laplace)

Qui invece **discretizziamo**, con una sola regola: ogni feature tagliata alla propria mediana.
Motivo: il Naive Bayes gaussiano, su questi dati, ha una varianza degenere
(`feature3_media` = 0,000180 nella classe 1, cinque valori su sei identici) che gli fa dominare il
prodotto con una densità di 26,91 contro 0,01–3,58 delle altre. Fuori campione perde 7,6 punti.

```python
P_sopra = (conteggio + 1) / (n_classe + 2)          # correzione di Laplace
punteggio_k = np.log(priori[k]) + Σ np.log(P(x_i | k))   # somma di logaritmi
```

**Risultato.** 1,0000 su `manuale.csv`, **0,7790** su `training.csv`. Tre celle su sedici avevano
stima grezza 0 o 1: senza Laplace, uno zero avrebbe azzerato un'intera classe.

**Dettaglio che rende credibile il 100%.** Il margine `|log P(1) − log P(0)|` va da **0,105 a
9,742**: un campione è classificato correttamente per un decimo di unità.

---

## 5. Task 3 — Data quality ed EDA

*Dettaglio completo in* `03_analisi_esplorativa.md`

**Il dataset non richiede pulizia.** Zero mancanti, `USERID` univoco, cinque vincoli logici su sei
soddisfatti.

**Il sesto vincolo fallisce ma non è un errore.** `n_giorni_attivi <= durata_giorni + 1` è falso in
392 righe (5,6%). Causa: `n_giorni_attivi` conta i **giorni di calendario** (`TIMESTAMP // 86400`),
quindi due azioni a cavallo della mezzanotte cadono in bucket diversi pur distando minuti. Verificato
sui timestamp grezzi dello studente 9.

**Nessuna riga va rimossa.** Le 358 righe con profilo identico sono studenti *diversi* con storie
brevissime (mediana 5 azioni contro 37): rimuoverle toglierebbe quasi solo abbandoni. I valori
estremi sono medie su decine di azioni reali.

![Distribuzione delle feature per classe](figure/eda_boxplot_per_classe.png)

Le quattro feature di quantità separano nettamente le classi; le `feature*_media` quasi per niente.

**Il risultato più importante dell'EDA:** le FEATURE del dataset **originale** sono standardizzate ma
quasi costanti.

![Concentrazione delle FEATURE grezze](figure/eda_feature_grezze.png)

Una sola causa spiega tre osservazioni fatte in task diversi: le correlazioni a zero (Task 1), la
varianza degenere del Naive Bayes gaussiano (Task 2), e un terzo di studenti bloccati sul valore
minimo di `feature0_media`. **Il difetto è del dataset originale, non del nostro preprocessing.**

![Matrice di correlazione](figure/eda_correlazione.png)

Due blocchi: le quattro feature di quantità correlate al target fra −0,55 e −0,58 e **fra loro fino
a 0,91**; le altre tre fra −0,03 e +0,08.

---

## 6. Task 4 — Ottimizzazione dei classificatori manuali

*Dettaglio completo in* `04_valutazione.md`

**Il protocollo è metà del contenuto.** Ricerca sul 70%, verifica sul 30% una volta sola,
cross-validation a 5 pieghe per il confronto finale — che serve soprattutto a fornire la
**deviazione standard**, pari a **0,0095**. Senza quella non si può dire se mezzo punto sia un
miglioramento o rumore.

**Decision tree.** Soglia da 22,5 a 28. Crescendo in profondità si guadagnano meno di 0,8 punti, ma
la deviazione standard cresce di oltre il 60% (0,0086 → 0,0141): overfitting. E su **29 nodi di
decisione, 16 (55%) hanno entrambi i rami che portano alla stessa classe** — riducono entropia senza
cambiare alcuna predizione.

**Naive Bayes — il risultato più istruttivo del progetto.** Il modello tarato su **12 campioni**
batteva quelli con soglie stimate su migliaia di righe. Invece di accettarlo, abbiamo formulato
un'ipotesi e l'abbiamo testata:

| soglie stimate da | accuratezza | confronto appaiato |
|---|---|---|
| `manuale.csv` (12 campioni, **bilanciato**) | 0,7902 | — |
| `training.csv` così com'è (57,7% positivi) | 0,7775 | perde **5 pieghe su 5** |
| `training.csv` **ribilanciato** | 0,7889 | pareggia (2/5) |

*(0,7902 e non 0,7790 perché qui dai 12 campioni vengono solo le **soglie**: le probabilità
condizionate sono ristimate su ogni piega di addestramento.)*

Il divario sparisce appena si ribilancia. Conclusione: per una soglia di taglio **non conta la
numerosità del campione, conta che rappresenti equamente le due classi**.

**Esito complessivo.** Da 0,7734 a 0,7933: +2,0 punti, circa due deviazioni standard.

---

## 7. Task 5 — Scikit-Learn e modello finale

*Dettaglio completo in* `05_modellazione.md`

**Protocollo.** Divisione 80/20 stratificata, test set **aperto solo alla fine**. Tutte le scelte in
cross-validation sul training set.

![Confronto dei classificatori](figure/ml_confronto_modelli.png)

Tre modelli su otto fanno **peggio** del classificatore manuale. Dopo il tuning i quattro
finalisti convergono fra 0,8081 e 0,8095: **1,4 millesimi**, contro una deviazione standard di 0,009.

**Due nostre ipotesi precedenti smentite dai dati** (e riportate lo stesso):

- **`RobustScaler` non serve**: dà lo stesso identico punteggio di `StandardScaler`;
- le tre feature «inerti» **contribuiscono**: toglierle peggiora tutti i modelli, per la regressione
  logistica in 5 pieghe su 5.

![Learning curve](figure/ml_learning_curve.png)

La regressione logistica ha le curve sovrapposte (divario **+0,0007**): regime di **bias**, nessun
overfitting. Ed è **piatta già da 450 campioni** — più dati non servirebbero. Il Gradient
Boosting parte con divario +0,0730 e arriva allo stesso punto.

**Modello scelto: regressione logistica**, `C = 0,1`, `StandardScaler`.

![Confusion matrix sul test set](figure/ml_confusion_matrix.png)

Non è stata scelta perché vince — i quattro finalisti stanno in 0,4 punti, meno di una deviazione
standard — ma perché a parità di prestazioni è la più semplice, non va in overfitting, ha
coefficienti leggibili ed è stabile.

**Il file d'esame.**

```python
import preprocessing as pp
X_esame, y_esame = pp.carica_per_predire("real_settings.csv")   # log di azioni O tabella aggregata
previsioni = finale.predict(X_esame)
```

Verificato in entrambi i formati: log di azioni (200 studenti, 0,8050) e già aggregato (0,8200).

---

## 8. I numeri da sapere a memoria

| | valore |
|---|---|
| azioni / studenti / attività | 411.749 / 7.047 / 97 |
| prevalenza a livello **azione** | 0,99% |
| prevalenza a livello **studente** | 57,71% |
| `manuale.csv` / `training.csv` | 12 (6+6) / 7.035 |
| numero di feature | 8 |
| baseline (sempre «abbandona») | 0,5771 |
| classificatori manuali su `manuale.csv` | 1,0000 entrambi |
| classificatori manuali su `training.csv` | decision tree 0,7734 — NB 0,7790 |
| miglior manuale dopo ottimizzazione (Task 4) | 0,7933 |
| miglior modello sklearn in CV | 0,8095 (tutti fra 0,8081 e 0,8095) |
| **modello finale sul test set** | **0,7925** (F1 0,8272, AUC 0,8485) |
| deviazione standard fra le pieghe | ≈ 0,009 |
| correlazione massima fra feature | 0,91 (`n_azioni` ↔ `n_attivita_distinte`) |
| `FEATURE3` costante nel | 93,5% delle azioni |

---

## 9. Come è scritto il codice — la vettorizzazione

**Cosa.** Nessun calcolo che tocchi i dati è scritto con un ciclo Python. Righe, colonne e soglie
candidate si attraversano con operazioni su array: confronti, maschere booleane, somme di colonna,
prodotti matriciali, `groupby`. I `for` che restano nei notebook non scorrono dati.

**Perché.** Tre motivi, in ordine di importanza.

1. **È la formula, scritta come si scrive.** Il Naive Bayes somma i logaritmi delle condizionate su
   tutte le feature: quella somma *è* un prodotto matriciale, e scriverla come `B @ log(P).T`
   avvicina il codice alla matematica invece di allontanarlo. Lo stesso vale per l'entropia: scritta
   sui conteggi diventa un'espressione aritmetica che vale per un nodo o per mille.
2. **Il ciclo non è dove si guarda quando si cerca un errore.** Un doppio ciclo su feature e soglie
   nasconde la logica dentro l'impalcatura che la fa girare. La versione vettorizzata dice in tre
   righe *cosa* si sta calcolando.
3. **La velocità, che cambia il modo di lavorare.** Il ciclo Python interpreta ogni singola
   operazione; NumPy e pandas la eseguono in C su un blocco contiguo di memoria. Non è comodità:
   un Task 4 che gira in 6 secondi invece di 68 lo si rilancia ogni volta che serve ricontrollare
   un numero, mentre uno lento spinge a fidarsi degli output già salvati — cioè al modo di
   lavorare sbagliato. E l'abitudine si prende **qui**, su 7.035 righe, dove costa poco, non sul
   dataset in cui il ciclo non finirebbe più: la dimensione di questo dataset è didattica, non un
   argomento per scrivere il codice peggio.

**Verificato.** Il Task 4 fa crescere 25 alberi (5 profondità × 5 pieghe), e ogni nodo valuta ~320
coppie (feature, soglia):

| | con i cicli | vettorizzato |
|---|---|---|
| `04_valutazione.ipynb`, notebook intero | 67,7 s | **6,6 s** |
| un albero di profondità 5 su 4.924 studenti | 4,7 s | **0,09 s** |

**Gli output non cambiano.** Le due versioni sono state confrontate riga per riga sugli stessi dati:
strutture degli alberi identiche, stesse predizioni su tutti i 7.035 studenti, scansioni di soglia
identiche **bit per bit**. Il Task 4 e il Task 5 producono output testuali identici al carattere.
L'unico scarto è nel Naive Bayes: sommare i logaritmi in ordine diverso cambia il risultato di
2 ulp (~3·10⁻¹⁵), cioè la sedicesima cifra decimale, senza toccare nessuna predizione.

**Le quattro trasformazioni principali.**

| dove | prima | dopo |
|---|---|---|
| NB, punteggi (2.1, 4) | due cicli annidati, classi × feature | `B @ log(P).T + (1-B) @ log(1-P).T` |
| NB, conteggi (4) | un sottoinsieme per ogni (feature, classe, valore) | un `np.bincount` sull'indice appiattito |
| decision tree, ricerca dello split (2.2, 4) | due cicli annidati, feature × soglie | una matrice `campioni × coppie candidate` |
| decision tree, predizione (2.2, 4) | `X.apply(scendi, axis=1)`, un percorso per riga | discesa a maschere, ricorsione sui **nodi** |

**I `for` che restano, e perché.**

| tipo | dove | perché non si vettorizza |
|---|---|---|
| pieghe di cross-validation | `cv_punteggi` (Task 4) | ogni piega **riaddestra** il modello |
| modelli e iperparametri | Task 5, quasi ovunque | ogni giro chiama `fit` su uno stimatore diverso |
| nomi di colonna | `classifica_feature`, `nb_quantili` | ogni feature ha un numero **diverso** di soglie, le liste non stanno in un unico array; il corpo del ciclo è già un'operazione su tutta la colonna |
| assi di un grafico | Task 3 e 5 | ogni giro disegna un oggetto matplotlib diverso |
| stampa di un riepilogo | qua e là | il calcolo è già fatto, resta solo da formattarlo |

La distinzione da tenere all'orale è questa: **si vettorizza ciò che ripete la stessa operazione su
dati diversi; non si vettorizza ciò che fa operazioni diverse.** Un ciclo su otto nomi di colonna in
cui ogni giro elabora 7.035 righe in blocco non è un ciclo sui dati.

---

## 10. Domande probabili all'orale

**Nel vostro codice non ci sono cicli sui dati: perché?**
Perché la formula si scrive meglio così. La somma dei logaritmi del Naive Bayes su tutte le feature
*è* un prodotto matriciale, e `B @ log(P).T` la dice più chiaramente di due cicli annidati.
L'entropia scritta sui conteggi invece che sulle etichette diventa un'espressione aritmetica che
NumPy applica a un nodo o a trecento split indifferentemente. La velocità viene dietro: il Task 4 è
passato da 67,7 a 6,6 secondi. Le due versioni sono state confrontate riga per riga e danno gli
stessi risultati — dettaglio nella sezione 9.

**Allora perché nel Task 5 ci sono ancora dei `for`?**
Perché lì ogni giro fa una cosa **diversa**: chiama `fit` su un modello diverso, o su una piega
diversa della cross-validation. Vettorizzare vuol dire applicare la *stessa* operazione a molti
dati in un colpo solo; dove l'operazione cambia a ogni giro non c'è niente da vettorizzare. La
regola che abbiamo seguito è: nessun ciclo Python che scorra righe, colonne o soglie candidate.

**Come fate a valutare tutte le soglie di un nodo senza un ciclo?**
Si impilano in un unico vettore tutte le coppie (feature, soglia) candidate e si costruisce la
matrice booleana `campioni × coppie`, dove la colonna *j* dice quali campioni finiscono nel ramo
sinistro dello split *j*. Da lì i conteggi delle due classi in ogni ramo sono due somme di colonna,
e l'information gain di tutti gli split esce da un'unica espressione. Il ciclo che resta scorre gli
otto *nomi* di colonna, perché ogni feature ha un numero diverso di soglie candidate.

**Perché avete cambiato l'unità di analisi?**
Perché `LABEL = 1` marca l'ultima azione di chi abbandona: descrive lo studente, non l'azione. A
livello azione il problema è quasi degenere — un classificatore che risponde sempre 0 fa 99,01% di
accuratezza, e la regola non-apprendente «è questa l'ultima azione?» fa 99,28% con recall 1,000.

**Il dataset è sbilanciato: perché non avete usato SMOTE o i pesi di classe?**
Perché non è sbilanciato. Lo 0,99% è la prevalenza nell'unità di analisi sbagliata; a livello
studente è 57,7%, cioè quasi bilanciato. Riequilibrare sarebbe stato curare un sintomo inesistente.

**Perché unire i file per posizione invece che con un `merge`?**
`ACTIONID` non è una chiave nel file delle etichette. Il `merge` restituisce lo stesso numero di
righe ma ne perde 15.116 e ne duplica altrettante: un controllo su `len()` non se ne accorgerebbe.

**Perché togliere l'ultima azione a tutti e non solo a chi abbandona?**
Perché altrimenti la regola di costruzione delle feature dipenderebbe dal target: sarebbe leakage.

**Perché `manuale.csv` ha 12 campioni estratti a caso e non scelti?**
La consegna chiede 10–15. Sono 6 + 6 per avere entropia iniziale 1 bit. A caso perché scegliere «i
casi più chiari» produrrebbe un file su cui qualunque classificatore funziona, rendendo la
valutazione priva di significato.

**Il vostro decision tree ha un solo nodo. Non è troppo poco?**
È l'algoritmo a fermarsi: `n_attivita_distinte <= 22,5` ha information gain 1,0000, entrambi i figli
sono puri e non c'è più entropia da ridurre. Non abbiamo applicato pruning né limitato la crescita.

**Un IG di 1,0000 è un buon segno?**
No, è un campanello d'allarme. Con 12 campioni e 8 feature continue è quasi sempre *possibile*
trovare un taglio che separa tutto. Infatti sul dataset completo quello stesso albero scende al
77,3%.

**Perché avete discretizzato per il Naive Bayes ma non per il decision tree?**
Sono problemi diversi. Per il decision tree le soglie sui punti medi sono l'algoritmo standard e il
confine lo sceglie l'information gain. Per il Naive Bayes la versione gaussiana, su questi dati, ha
una varianza degenere (0,000180) che fa dominare il prodotto a una sola feature; discretizzare alla
mediana usa una regola sola, uguale per tutte, e generalizza meglio di 7,6 punti.

**A cosa serve la correzione di Laplace?**
A evitare che una probabilità condizionata stimata a zero azzeri l'intero prodotto. Non è teorico:
tre celle su sedici avevano stima grezza 0 o 1.

**Perché sommate logaritmi invece di moltiplicare probabilità?**
Il prodotto di otto probabilità è piccolissimo e rischia l'underflow. Il logaritmo è monotono,
quindi la classe che massimizza la somma è la stessa che massimizza il prodotto.

**L'ipotesi di indipendenza del Naive Bayes è rispettata?**
No, ed è violata in modo grave: quattro feature misurano la lunghezza della storia e sono correlate
fino a 0,91. Il modello conta più volte la stessa evidenza, quindi le sue probabilità sono **mal
calibrate**. Danneggia poco la *decisione*, perché conta solo quale punteggio sia maggiore.

**Avete trovato osservazioni errate nel dataset?**
No. Un solo vincolo logico fallisce, `n_giorni_attivi <= durata_giorni + 1`, ma è corretto:
«giorno» è un giorno di calendario, quindi due azioni a cavallo della mezzanotte contano come due
giorni pur distando minuti.

**Perché non avete rimosso duplicati e outlier?**
I 358 profili identici sono studenti diversi con storie brevissime — rimuoverli toglierebbe quasi
solo abbandoni e distorcerebbe il target. Gli outlier sono medie su decine di azioni reali, non
refusi.

**Perché lo scaling sta dentro la `Pipeline`?**
Perché così media e deviazione standard vengono stimate solo sulle pieghe di addestramento. Stimarle
su tutto il dataset farebbe filtrare informazione dalla piega di validazione.

**Avete scelto la regressione logistica: non è il modello meno potente?**
Sì, ed è il punto. I quattro finalisti stanno in 0,4 punti sul test set, meno di una deviazione
standard: non c'è un vincitore statistico. A parità di prestazioni si sceglie sul resto — nessun
overfitting (divario +0,0007), un solo iperparametro, coefficienti interpretabili.

**Cosa vi dicono le learning curve?**
Che la regressione logistica è in regime di **bias**: curve sovrapposte, nessuna varianza. E che la
curva di validazione è piatta già da 450 campioni, quindi più dati non aiuterebbero.

**Perché nessun modello supera il 79-81%?**
Tre indizi convergono: tutti i modelli si fermano lì; la learning curve è piatta; e le
feature contengono poca informazione — tre su otto sono quasi costanti nel dataset originale, le
altre cinque sono correlate fino a 0,91. Il limite è nei dati, non nel modello.

**Come applicherete il modello a `real_settings.csv`?**
Le otto feature sono definite da noi e non esistono nel dataset originale, quindi il file va
trasformato prima di darlo al modello. `preprocessing.py` lo fa con la stessa identica funzione
usata in addestramento, e accetta entrambi i formati possibili: log di azioni o tabella già
aggregata. Testato su tutti e due.

**Perché il modello non è salvato su disco?**
L'addestramento dura meno di un secondo: rieseguire il notebook evita ogni problema di compatibilità
fra versioni di Scikit-Learn nel caricare un oggetto serializzato.

**Qual è il limite del vostro lavoro?**
La relazione «poca attività → abbandono» è in parte **tautologica**: un abbandono è per definizione
la fine dell'attività. Applicando lo stesso modello a una finestra iniziale di osservazione, l'AUC
crolla a 0,535. Il 79% è un ottimo risultato sul problema *come è formulato dal dataset*, non una
previsione dell'abbandono a corso in corso.

---

## 11. I limiti che dichiariamo per primi

Dichiararli prima che li trovi il docente è la parte più importante dell'esposizione.

1. **La tautologia.** Le feature descrivono l'intera storia dello studente, che per un abbandono
   finisce per definizione. Verificato in appendice al Task 1: su una finestra iniziale l'AUC scende
   a 0,535.
2. **Collinearità fino a 0,91.** I coefficienti del modello finale non sono interpretabili
   singolarmente.
3. **Tre feature su otto quasi prive di segnale**, per un difetto del dataset originale. Le teniamo
   perché in combinazione contribuiscono, ma non ci si deve aspettare nulla da loro.
4. **La finestra di osservazione è di 29,77 giorni.** `n_azioni` e `durata_giorni` dipendono da
   quella ampiezza: un file d'esame con una finestra diversa produrrebbe feature su scala diversa.
   Misurato nel Task 5: su storie complete il modello fa 0,7833, su una fetta del log 0,05–0,22.
   `preprocessing.py` avvisa quando la mediana di `n_azioni` è meno di un terzo di quella di
   `training.csv`.
5. **Un suggerimento del Task 3 si è rivelato sbagliato** (`RobustScaler`): l'abbiamo verificato
   e riportato invece di rimuoverlo.

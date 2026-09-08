# Task 5 — Classificatori Scikit-Learn e scelta del modello finale

**Obiettivo.** Addestrare più classificatori con Scikit-Learn separando training e test set,
massimizzare le prestazioni sul test set, analizzare criticamente le decisioni e selezionare il
classificatore finale.

**Risultato.** Il modello scelto è una **regressione logistica** (`StandardScaler`, `C = 0,1`):
**79,25%** di accuratezza sul test set, F1 0,8272, ROC-AUC 0,8485, contro una baseline del 57,71%.

---

## 1. Protocollo

**Cosa.** Divisione 80/20 stratificata di `training.csv` (5.628 / 1.407 studenti), con il test set
messo da parte **subito** e non toccato fino alla sezione 7. Ogni scelta — modelli, scaler, feature,
iperparametri, soglia — è fatta in 5-fold cross-validation sul solo training set.

**Perché.** È la regola che il Task 4 ha mostrato necessaria: scegliere e misurare sugli stessi dati
restituisce il massimo di un insieme di rumore, non una stima delle prestazioni.

| riferimento | accuratezza |
|---|---|
| baseline: rispondere sempre «abbandona» | 0,5771 |
| miglior classificatore **manuale** (Task 4) | 0,7933 |

Il secondo è il più interessante: dice quanto valga usare Scikit-Learn invece delle venti righe del
Task 2.

---

## 2. I classificatori in gara

I **cinque modelli visti a lezione**, con **parametri di default**, per avere un punto di partenza
non influenzato dalle nostre scelte. Il k-NN parte dal k predefinito di Scikit-Learn, 5.

**Lo scaling dentro una `Pipeline`.** I modelli basati su distanze o gradienti lo richiedono, quelli
ad albero no. Metterlo nella pipeline garantisce che media e deviazione standard siano stimate sulle
sole pieghe di addestramento — la regola 2 del progetto.

| modello | accuratezza | dev.std | F1 | ROC-AUC |
|---|---|---|---|---|
| Regressione logistica | **0,8074** | 0,0095 | 0,8389 | 0,8683 |
| Naive Bayes gaussiano | 0,7944 | 0,0086 | 0,8232 | 0,8610 |
| k-NN (k=5) | 0,7839 | 0,0073 | 0,8173 | 0,8310 |
| Perceptron | 0,7228 | 0,0683 | 0,7691 | 0,7763 |
| Decision tree | 0,7178 | 0,0191 | 0,7537 | 0,7108 |

**Tre letture.** Il **decision tree senza limiti di profondità** è il peggiore di tutti (0,7178),
sotto perfino al Perceptron: cresce fino a foglie pure e impara il rumore. Il **Perceptron** ha una
deviazione standard di 0,0683, sette volte quella della regressione logistica: i dati non sono
linearmente separabili e la soluzione dipende dalla piega. E **tre modelli su cinque fanno peggio
del classificatore manuale del Task 4**: solo la regressione logistica lo supera, il Naive Bayes lo
eguaglia. Scikit-Learn non regala niente.

---

## 3. Le due domande lasciate aperte dal Task 3

### 3.1 `RobustScaler` non serve

| modello | StandardScaler | RobustScaler | QuantileTransformer |
|---|---|---|---|
| Regressione logistica | 0,8074 | 0,8074 | 0,7987 |
| k-NN | 0,7839 | 0,7816 | 0,7845 |

**Il suggerimento del Task 3 era sbagliato, e lo diciamo.** Le code pesanti avrebbero dovuto
penalizzare la standardizzazione: falso su questi dati, perché gli outlier sono poche decine su
7.035. `RobustScaler` dà alla regressione logistica lo stesso identico punteggio e al k-NN due
millesimi in meno. Restiamo su `StandardScaler`.

I due modelli provati sono quelli a cui lo scaling serve davvero — uno basato sul gradiente, uno
sulle distanze. Il decision tree non compare perché è invariante a trasformazioni monotone delle
feature: per lui lo scaler non cambia nulla per costruzione, non per un fatto sperimentale.

### 3.2 Le feature «inerti»: la risposta dipende dal modello

| modello | 8 feature | 5 feature | differenza | pieghe a favore di 8 |
|---|---|---|---|---|
| Regressione logistica | 0,8074 | 0,8047 | +0,0027 | **5/5** |
| k-NN | 0,7839 | 0,7850 | −0,0011 | 2/5 |
| Decision tree | 0,7178 | 0,7168 | +0,0011 | 3/5 |

**Il risultato non è uniforme, e riportarlo così com'è è parte della risposta.** Per la regressione
logistica — il modello che verrà scelto — togliere le tre feature peggiora il punteggio in **5 pieghe
su 5**: piccolo, ma sistematico. Per k-NN e decision tree la differenza è di un millesimo scarso e
cambia segno da una piega all'altra: è rumore, non un effetto.

La lettura corretta è che le tre feature non sono individualmente correlate col target (fra −0,03 e
+0,08) ma **in combinazione lineare** aggiungono qualcosa, e che un modello lineare è proprio quello
in grado di sfruttarle. Nessuna delle due letture autorizza a scartarle: un effetto piccolo ma
sistematico non è un effetto nullo, e **scartare feature guardando la sola correlazione resta
rischioso**.

---

## 4. Ottimizzazione degli iperparametri

| modello | prima | dopo | guadagno | parametri scelti |
|---|---|---|---|---|
| Regressione logistica | 0,8074 | **0,8081** | +0,0007 | `C=0,1` |
| k-NN | 0,7839 | 0,8079 | **+0,0240** | `n_neighbors=51`, `weights='uniform'` |
| Decision tree | 0,7178 | 0,7934 | **+0,0755** | `max_depth=5`, `min_samples_leaf=20` |

Il Naive Bayes gaussiano non compare perché non ha iperparametri da tarare: entra in finale nella sua
configurazione di default. Il Perceptron è escluso dopo il risultato della sezione 2.

**L'ottimizzazione conta soprattutto per i modelli peggiori.** Il decision tree guadagna 7,6 punti e
il k-NN 2,4: sono i due che di default andavano in overfitting, ciascuno a modo suo — l'albero
costruendo foglie da un solo studente, il k-NN guardando un intorno di 5 vicini in uno spazio
rumoroso. Entrambi si correggono nello stesso modo, **togliendo capacità al modello**: potando
l'albero a profondità 5 con foglie da almeno 20 studenti, allargando l'intorno del k-NN da 5 a 51
vicini.

La regressione logistica guadagna **sette decimillesimi**: era già al suo massimo.

**Il risultato importante è però la convergenza.** I due migliori arrivano allo stesso punto da
strade opposte: 0,8081 e 0,8079, **due decimillesimi**, contro una deviazione standard di circa
0,009. Sono statisticamente indistinguibili. Quando un confine lineare e 51 vicini danno lo stesso
numero, il limite non è nel modello ma **nei dati**.

---

## 5. La soglia di decisione

**Verificato.** Sull'intervallo 0,30–0,70 l'accuratezza è massima esattamente a **0,50** (0,8081),
mentre l'F1 sarebbe massimo a **0,40** (0,8432).

**Cosa abbiamo scelto.** Manteniamo 0,5: la consegna chiede di massimizzare le prestazioni senza
indicare un costo asimmetrico fra i due tipi di errore. Se l'obiettivo fosse intercettare più
abbandoni possibile — lo scenario realistico per un intervento didattico — la soglia andrebbe
abbassata a 0,40, accettando più falsi allarmi.

---

## 6. Analisi bias-varianza

Il **divario** fra curva di addestramento e curva di validazione misura la varianza; il livello a cui
si stabilizzano misura il bias.

| modello | divario finale | regime |
|---|---|---|
| Regressione logistica | **+0,0007** | bias: nessun overfitting |
| Decision tree, default | **+0,2803** | varianza pura: memorizza il training set |
| Decision tree, potato (sez. 4) | +0,0187 | la potatura chiude il divario |

**I due estremi, misurati sullo stesso dataset.** La regressione logistica ha le curve
*sovrapposte*: impara tutto quello che le feature contengono e nient'altro. Il decision tree
lasciato crescere sta all'opposto: quasi il 100% di accuratezza in addestramento contro il 72% in
validazione, cioè memorizza gli studenti invece di generalizzare.

Il confronto fra le due righe dell'albero è la dimostrazione pratica del compromesso: la potatura
della sezione 4 chiude il divario da +0,2803 a +0,0187 e alza la validazione da 0,7173 a 0,7934.
**7,6 punti guadagnati non aggiungendo dati o feature, ma togliendo capacità al modello.**

**Il dettaglio decisivo.** La curva di validazione della regressione logistica è **piatta già da 450
campioni** (0,8061 a 450, 0,8081 a 4.502). Con dieci volte gli studenti il risultato sarebbe lo
stesso: il limite sono le **feature**, come previsto dai Task 3 e 4.

---

## 7. Il modello finale

| modello | CV | test: accuratezza | test: F1 | test: ROC-AUC |
|---|---|---|---|---|
| **Regressione logistica** | **0,8081** | **0,7925** | **0,8272** | 0,8485 |
| k-NN | 0,8079 | 0,7875 | 0,8240 | 0,8530 |
| Naive Bayes gaussiano | 0,7944 | 0,7790 | 0,8112 | 0,8408 |
| Decision tree | 0,7934 | 0,7747 | 0,8143 | 0,8404 |

**La scelta: regressione logistica**, `C = 0,1`, `StandardScaler`.

È prima in **entrambe** le valutazioni, in cross-validation e sul test set. Il punto delicato è che
il k-NN ottimizzato le sta a due decimillesimi in CV e a mezzo punto sul test set: fra i due non c'è
un vincitore statistico, ed è corretto dirlo invece di rivendicare un primato che i numeri non
sostengono. Quando i punteggi sono equivalenti la scelta si fa su altri criteri:

| criterio | perché |
|---|---|
| prestazioni | la migliore sia in CV sia sul test set, seppure di poco |
| assenza di overfitting | learning curve sovrapposte (+0,0007) |
| semplicità | un iperparametro contro i due del k-NN |
| interpretabilità | coefficienti leggibili; il k-NN non ne ha |
| costo di predizione | il k-NN deve tenere in memoria tutto il training set e calcolare 5.628 distanze per ogni studente |

Il criterio è stato fissato **prima** di aprire il test set: che la regressione logistica risulti
prima anche lì è una conferma, non la motivazione della scelta.

**Prestazioni dettagliate.** Precision 0,7961 e recall 0,8608 sugli abbandoni; 0,7864 e 0,6992 su
chi prosegue. Il modello sbaglia più spesso classificando come abbandono chi prosegue (179 casi) che
il contrario (113).

**I coefficienti** (su feature standardizzate, quindi confrontabili):

| feature | coefficiente |
|---|---|
| `n_attivita_distinte` | −0,645 |
| `n_giorni_attivi` | −0,644 |
| `n_azioni` | −0,312 |
| `durata_giorni` | −0,308 |
| `feature0_media` | −0,299 |
| `feature3_media` | +0,203 |
| `feature1_media` | +0,141 |
| `feature2_media` | +0,038 |

Le prime cinque hanno segno **negativo**: più attività, giorni, azioni e durata, meno probabilità di
abbandono. È la stessa relazione trovata nel Task 2.

**Con l'avvertimento del Task 3.** Con correlazioni fino a 0,91 i coefficienti **non sono
interpretabili singolarmente**: che `n_attivita_distinte` pesi −0,645 e `n_azioni` −0,312 non
significa che la prima conti il doppio, ma che il modello ha distribuito fra due variabili quasi
identiche un'unica quantità di informazione.

---

## 8. Il file d'esame `real_settings.csv`

Le otto feature non esistono nel dataset originale, quindi `real_settings.csv` va trasformato prima
di poterlo dare al modello. `preprocessing.py` lo fa con la stessa identica funzione usata in
addestramento e accetta entrambi i formati possibili.

**Verificato end-to-end** su un file finto di 200 studenti: log di azioni → 200 studenti, colonne
corrette, accuratezza 0,8050; tabella già a livello studente → 50 studenti, accuratezza 0,8200.

**Il caso da temere: un file che è solo una fetta del log.** Le feature descrivono la storia
*completa* di uno studente. Se il file d'esame ne contenesse un frammento, le storie sarebbero
troncate e il modello risponderebbe «abbandona» quasi a tutti:

| contenuto del file d'esame | studenti | previsti «abbandona» | accuratezza |
|---|---|---|---|
| storie **complete** di 300 studenti | 300 | 61% | **0,7833** |
| 3.000 azioni estratte a caso | 624 | 95% | 0,0625 |
| prime 3.000 righe del log | 309 | 100% | 0,0453 |
| azioni dei primi 7 giorni | 3.776 | 100% | 0,2238 |

`preprocessing.py` intercetta i primi due casi con un avviso, quando la mediana di `n_azioni` scende
sotto un terzo di quella di `training.csv`. Il terzo sfugge: una finestra temporale iniziale produce
storie brevi ma verosimili. È il limite 4 del Task 1, qui misurato invece che solo dichiarato.

```python
import preprocessing as pp
X_esame, y_esame = pp.carica_per_predire("real_settings.csv")
previsioni = finale.predict(X_esame)
```

**Il modello non è salvato su disco**: l'addestramento dura meno di un secondo, e rieseguire il
notebook evita ogni problema di compatibilità fra versioni di Scikit-Learn.

---

## 9. Conclusioni

**Quanto è servito Scikit-Learn.** Il miglior classificatore manuale arrivava a 0,7933 in
cross-validation, contro 0,8081 del modello finale: **+1,5 punti**, circa una deviazione standard e
mezza. Un guadagno reale ma piccolo, e dirlo è più onesto che presentarlo come una svolta.

**Perché il tetto è dove è.** Tre indizi indipendenti convergono:

1. i due modelli migliori, partendo da famiglie opposte, si fermano entrambi a 0,808;
2. la learning curve è piatta già da 450 campioni;
3. tre feature su otto sono quasi costanti nel dataset **originale** (`FEATURE3` ha lo stesso valore
   nel 93,5% delle azioni) e le altre cinque sono correlate fino a 0,91.

Il limite è **quanta informazione le feature contengono**, non il modello né la quantità di dati.

**Limite dichiarato.** La relazione «poca attività → abbandono» è in parte **tautologica**: un
abbandono è per definizione la fine dell'attività. Su una finestra iniziale di osservazione l'AUC
crolla a 0,535 (appendice del Task 1). Il 79% vale sul problema *come è formulato dal dataset*, non
come previsione dell'abbandono a corso in corso.

---

## Riepilogo

| passo | risultato |
|---|---|
| protocollo | 80/20 stratificato, test set aperto solo alla fine |
| modelli confrontati | i 5 visti a lezione, con parametri di default, più baseline |
| scaler | `StandardScaler`; `RobustScaler` non cambia nulla |
| feature | tutte e 8; toglierne 3 peggiora la regressione logistica in 5 pieghe su 5 |
| tuning | +0,0755 il decision tree, +0,0240 il k-NN, +0,0007 la regressione logistica |
| soglia di decisione | 0,5 |
| bias-varianza | regressione logistica in regime di bias (+0,0007), albero non potato in varianza (+0,2803) |
| **modello finale** | **regressione logistica, `C = 0,1`, `StandardScaler`** |
| prestazioni sul test set | accuratezza 0,7925, F1 0,8272, ROC-AUC 0,8485 |
| guadagno sui classificatori manuali | +1,5 punti |

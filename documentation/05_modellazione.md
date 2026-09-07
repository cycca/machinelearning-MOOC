# Task 5 — Classificatori Scikit-Learn e scelta del modello finale

**Obiettivo.** Addestrare più classificatori con Scikit-Learn separando training e test set,
massimizzare le prestazioni sul test set, analizzare criticamente le decisioni e selezionare il
classificatore finale — quello che in sede d'esame verrà applicato a `real_settings.csv`.

**Prodotti.** `05_modellazione.ipynb` (13 celle di codice, 4 figure).

**Risultato.** Il modello scelto è una **regressione logistica** (`StandardScaler`, `C = 0,1`):
**79,25%** di accuratezza sul test set, F1 0,8272, ROC-AUC 0,8485, contro una baseline del 57,71%.

---

## 1. Protocollo

**Cosa abbiamo fatto.** Divisione 80/20 stratificata di `training.csv` (5.628 / 1.407 studenti), con
il test set messo da parte **subito** e non toccato fino alla sezione 7. Tutte le scelte — modelli,
scaler, feature, iperparametri, soglia — sono state fatte in 5-fold cross-validation sul solo
training set.

**Perché.** È la regola che il Task 4 aveva mostrato essere necessaria: scegliere e misurare sugli
stessi dati restituisce il massimo di un insieme di rumore, non una stima delle prestazioni.
La stratificazione conserva il 57,7% di positivi in entrambe le parti.

**I due riferimenti.** Ogni risultato va letto contro:

| riferimento | accuratezza |
|---|---|
| baseline: rispondere sempre «abbandona» | 0,5771 |
| miglior classificatore **costruito a mano** (Task 4) | 0,7933 |

Il secondo è il più interessante: dice quanto valga davvero usare Scikit-Learn invece di venti righe
scritte a mano.

---

## 2. I classificatori in gara

**Cosa abbiamo fatto.** Otto modelli visti a lezione, tutti con **parametri di default**, per avere
un punto di partenza non influenzato dalle nostre scelte.

**Lo scaling dentro una `Pipeline`.** I modelli basati su distanze o gradienti (k-NN, SVC,
perceptron, regressione logistica) richiedono feature confrontabili; quelli ad albero no. Metterlo
nella pipeline garantisce che media e deviazione standard siano stimate **sulle sole pieghe di
addestramento**, mai sulla piega di validazione — la regola 2 del progetto.

**Cosa abbiamo verificato.**

| modello | accuratezza | dev.std | F1 | ROC-AUC |
|---|---|---|---|---|
| Regressione logistica | **0,8074** | 0,0095 | 0,8389 | 0,8683 |
| SVC (RBF) | 0,8072 | 0,0065 | 0,8401 | 0,8416 |
| Gradient Boosting | 0,8017 | 0,0056 | 0,8333 | 0,8585 |
| Random Forest | 0,7998 | 0,0093 | 0,8305 | 0,8628 |
| Naive Bayes gaussiano | 0,7944 | 0,0086 | 0,8232 | 0,8610 |
| k-NN (k=5) | 0,7839 | 0,0073 | 0,8173 | 0,8310 |
| Perceptron | 0,7228 | 0,0683 | 0,7691 | 0,7763 |
| Albero di decisione | 0,7178 | 0,0191 | 0,7537 | 0,7108 |

**Tre letture.** L'**albero senza limiti di profondità** è il peggiore: cresce fino a foglie pure e
impara il rumore — la stessa lezione del Task 4, dove metà dei nodi non cambiava alcuna predizione.
Il **Perceptron** ha una deviazione standard di 0,068, dieci volte quella degli altri: i dati non
sono linearmente separabili e l'algoritmo non converge a una soluzione stabile.

Soprattutto: **tre modelli su otto fanno peggio del classificatore costruito a mano** e un quarto lo
eguaglia. Il salto vero rispetto al Task 4 è di poco più di un punto.

---

## 3. Le due domande lasciate aperte dal Task 3

### 3.1 Lo scaler robusto non serve

**Cosa abbiamo verificato.**

| modello | StandardScaler | RobustScaler | QuantileTransformer |
|---|---|---|---|
| Regressione logistica | 0,8074 | 0,8074 | 0,7987 |
| SVC (RBF) | 0,8072 | 0,8081 | 0,8060 |

**Il suggerimento del Task 3 era sbagliato, e lo diciamo.** Avevamo scritto che le code pesanti
avrebbero dovuto penalizzare la standardizzazione. Ragionevole, ma falso su questi dati: gli outlier
sono poche decine su 7.035, troppo pochi per spostare media e deviazione standard in modo dannoso.
Restiamo su `StandardScaler`.

### 3.2 Le feature «inerti» contribuiscono comunque

**Cosa abbiamo verificato.**

| modello | 8 feature | 5 feature | differenza | pieghe a favore di 8 |
|---|---|---|---|---|
| Regressione logistica | 0,8074 | 0,8047 | +0,0027 | **5/5** |
| Random Forest | 0,7998 | 0,7916 | +0,0082 | 3/5 |
| Gradient Boosting | 0,8017 | 0,7969 | +0,0048 | 3/5 |

**Perché conta.** Togliere `feature1_media`, `feature2_media` e `feature3_media` peggiora tutti e
tre i modelli. Non sono individualmente correlate col target (fra −0,03 e +0,08), ma **in
combinazione** aggiungono qualcosa. Conferma che la scelta del Task 3 — tenerle — era corretta, e
mostra perché scartare feature guardando solo la correlazione singola sia rischioso.

---

## 4. Ottimizzazione degli iperparametri

**Cosa abbiamo fatto.** `GridSearchCV` con la stessa cross-validation, sui quattro modelli più
promettenti.

**Cosa abbiamo verificato.**

| modello | prima | dopo | guadagno | parametri scelti |
|---|---|---|---|---|
| Gradient Boosting | 0,8017 | **0,8095** | +0,0078 | `learning_rate=0,05`, `max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=50` |
| SVC (RBF) | 0,8072 | 0,8092 | +0,0020 | `C=1`, `gamma=0,05` |
| Random Forest | 0,7998 | 0,8092 | +0,0094 | `n_estimators=300`, `min_samples_leaf=20` |
| Regressione logistica | 0,8074 | 0,8081 | +0,0007 | `C=0,1` |

**Il risultato importante è la convergenza.** I quattro modelli finiscono fra 0,8081 e 0,8095:
**1,4 millesimi di distanza**, contro una deviazione standard fra le pieghe di circa 0,009. Sono
statisticamente indistinguibili.

Non è una delusione, è un'informazione: quando modelli con capacità molto diverse — un confine
lineare, un kernel RBF, 300 alberi, un boosting — arrivano tutti allo stesso punto, il limite non è
nel modello ma **nei dati**.

*(Nota tecnica: i modelli d'insieme sono definiti senza `n_jobs`, lasciando la parallelizzazione
alla cross-validation esterna. Annidare due livelli di parallelismo provoca oversubscription della
CPU e un warning di Scikit-Learn.)*

---

## 5. La soglia di decisione

**Cosa abbiamo fatto.** Un classificatore probabilistico decide confrontando `P(abbandono)` con 0,5.
Con classi non perfettamente bilanciate quella soglia non è necessariamente la migliore, ed è il
modo più economico per spostare l'equilibrio fra precisione e richiamo.

**Cosa abbiamo verificato.** Sull'intervallo 0,30–0,70, l'accuratezza è massima esattamente a
**0,50** (0,8081), mentre l'F1 sarebbe massimo a **0,40** (0,8432).

**Cosa abbiamo scelto.** Manteniamo 0,5. La consegna chiede di massimizzare le prestazioni senza
indicare un costo asimmetrico fra i due tipi di errore: in assenza di quel costo non c'è motivo di
privilegiare una classe. Se l'obiettivo fosse *intercettare* più abbandoni possibile — lo scenario
realistico per un intervento didattico — la soglia andrebbe abbassata a 0,40, accettando più falsi
allarmi.

---

## 6. Analisi bias-varianza

**Cosa abbiamo fatto.** Curve di apprendimento per il modello più semplice e per il più complesso.
Il **divario** fra la curva di addestramento e quella di validazione misura la varianza; il livello
a cui si stabilizzano misura il bias.

**Cosa abbiamo verificato.**

| modello | divario finale | regime |
|---|---|---|
| Regressione logistica | **+0,0007** | bias: nessun sovradattamento |
| Gradient Boosting | +0,0242 | varianza: memorizza, poi recupera con più dati |

**Il dettaglio decisivo.** La curva di validazione della regressione logistica è **piatta già da 450
campioni** (0,8061 a 450, 0,8081 a 4.502). Più dati non servirebbero: con dieci volte gli studenti
il risultato sarebbe lo stesso. Il limite sono le **feature**, esattamente come previsto dai
Task 3 e 4.

Il Gradient Boosting parte con un divario ampio e lo riduce all'aumentare dei dati, ma la sua curva
di validazione **arriva allo stesso punto**: più capacità non compra più prestazioni.

---

## 7. Il modello finale

**Cosa abbiamo verificato.** Solo a scelte fatte abbiamo aperto il test set:

| modello | CV | test: accuratezza | test: F1 | test: ROC-AUC |
|---|---|---|---|---|
| **Regressione logistica** | 0,8081 | **0,7925** | **0,8272** | 0,8485 |
| SVC (RBF) | 0,8092 | 0,7903 | 0,8255 | 0,8227 |
| Gradient Boosting | 0,8095 | 0,7903 | 0,8255 | 0,8520 |
| Random Forest | 0,8092 | 0,7882 | 0,8228 | 0,8552 |

**La scelta: regressione logistica** con `StandardScaler` e `C = 0,1`.

Le quattro alternative sono separate da meno di mezzo punto sul test set, cioè **meno di una
deviazione standard**: non c'è un vincitore statistico. Quando i punteggi sono equivalenti la scelta
si fa su altri criteri, e la regressione logistica li soddisfa tutti:

| criterio | perché |
|---|---|
| prestazioni | la migliore sul test set, seppure di poco |
| assenza di sovradattamento | curve di apprendimento sovrapposte (divario +0,0007) |
| semplicità | un iperparametro contro i quattro del boosting |
| interpretabilità | coefficienti leggibili |
| stabilità all'esame | nessuna dipendenza da seed o da algoritmi complessi |

**Prestazioni dettagliate sul test set.** Precisione 0,7961 e richiamo 0,8608 sugli abbandoni;
precisione 0,7864 e richiamo 0,6992 su chi prosegue. Il modello sbaglia più spesso classificando
come abbandono chi invece prosegue (179 casi) che il contrario (113).

**I coefficienti.** Su feature standardizzate, quindi confrontabili:

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
abbandono. È la stessa relazione trovata a mano nel Task 2, ora pesata su tutte le variabili
insieme.

**Con l'avvertimento del Task 3.** Con correlazioni fino a 0,91 fra le feature, i coefficienti **non
sono interpretabili singolarmente**: che `n_attivita_distinte` pesi −0,645 e `n_azioni` −0,312 non
significa che la prima conti il doppio, ma che il modello ha distribuito fra due variabili quasi
identiche un'unica quantità di informazione.

---

## 8. Il file d'esame `real_settings.csv`

**Cosa abbiamo fatto.** Verificato end-to-end la procedura su un file finto, costruito con 200
studenti scelti a caso dal registro grezzo, in **entrambi** i formati possibili.

**Cosa abbiamo verificato.** Log di azioni → 200 studenti, colonne corrette, accuratezza 0,8050.
File già a livello studente → 50 studenti, colonne corrette, accuratezza 0,8200.

```python
import preprocessing as pp
X_esame, y_esame = pp.carica_per_predire("real_settings.csv")
previsioni = finale.predict(X_esame)
```

**Perché il modello non è salvato su disco.** L'addestramento dura meno di un secondo: rieseguire il
notebook evita ogni problema di compatibilità fra versioni di Scikit-Learn nel caricare un file
serializzato.

---

## 9. Conclusioni

**Quanto è servito Scikit-Learn.** Il miglior classificatore costruito a mano nel Task 4 arrivava a
0,7933 in cross-validation, contro 0,8081 del modello finale: **+1,5 punti**, circa una deviazione
standard e mezza. Un guadagno reale ma piccolo, e dirlo è più onesto che presentare il modello
finale come una svolta.

**Perché il tetto è dove è.** Tre indizi indipendenti convergono:

1. tutti i modelli, dal confine lineare al boosting, si fermano a 0,809;
2. la curva di apprendimento è piatta già da 450 campioni;
3. il Task 3 aveva mostrato che tre feature su otto sono quasi costanti nel dataset **originale**
   (`FEATURE3` ha lo stesso valore nel 93,5% delle azioni) e che le altre cinque sono correlate fino
   a 0,91.

Il limite non è il modello né la quantità di dati: è **quanta informazione le feature contengono**.

**Il limite onesto da dichiarare.** Vale qui come nei Task 1, 2 e 4: la relazione «poca attività →
abbandono» è in parte **tautologica**, perché un abbandono è per definizione la fine dell'attività.
Un modello utile in pratica andrebbe addestrato su una **finestra iniziale** di osservazione — e
l'appendice del Task 1 mostra che in quel caso l'AUC crolla a 0,535. Il 79% va letto per quello che
è: un ottimo risultato sul problema **come è formulato dal dataset**, non una previsione
dell'abbandono a corso in corso.

---

## Riepilogo

| passo | risultato |
|---|---|
| protocollo | 80/20 stratificato, test set aperto solo alla fine; 5-fold CV per ogni scelta |
| modelli confrontati | 8 con parametri di default, più baseline |
| scaler | `StandardScaler`; il robusto non cambia nulla |
| feature | tutte e 8; toglierne 3 peggiora tutti i modelli |
| tuning | guadagni fra +0,0007 e +0,0094; tutti i modelli convergono a ~0,809 |
| soglia di decisione | 0,5 (ottima per l'accuratezza; 0,40 lo sarebbe per l'F1) |
| bias-varianza | regressione logistica in regime di bias, divario +0,0007 |
| **modello finale** | **regressione logistica, `C = 0,1`, `StandardScaler`** |
| prestazioni sul test set | accuratezza 0,7925, F1 0,8272, ROC-AUC 0,8485 |
| guadagno sui classificatori manuali | +1,5 punti |
| file d'esame | verificato in entrambi i formati via `preprocessing.py` |

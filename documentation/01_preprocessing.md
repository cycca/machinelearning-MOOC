# Task 1 — Preprocessing e preparazione dei dati

**Obiettivo.** Trasformare *act-mooc* in un formato leggibile da un classificatore ed estrarre
`manuale.csv` (12 campioni) e `training.csv` (7.035 campioni).

**Risultato.** 7.047 righe-studente, 8 feature, target `ABBANDONO` al 57,7% di positivi, zero
valori mancanti.

---

## 0. Il dataset

*act-mooc* (Stanford SNAP): 411.749 azioni di 7.047 studenti su 97 attività, in 29,77 giorni.
Distribuito in tre file TSV con lo stesso numero di righe, nello stesso ordine:

| file | colonne |
|---|---|
| `mooc_actions.tsv` | `ACTIONID`, `USERID`, `TARGETID`, `TIMESTAMP` |
| `mooc_action_features.tsv` | `ACTIONID`, `FEATURE0..3` |
| `mooc_action_labels.tsv` | `ACTIONID`, `LABEL` (1 sull'azione dopo cui lo studente abbandona) |

Non è un dataset tabellare ma una sequenza di eventi. Il lavoro del Task 1 non è pulire i dati —
sono già puliti — ma **cambiare l'unità di analisi**, dall'azione allo studente.

---

## 1. Unione dei tre file

**Cosa.** Un'unica tabella 411.749 × 9, unendo le colonne **per posizione** con `.to_numpy()`.

**Perché.** `ACTIONID` è una chiave vera solo in due file su tre:

| file | righe | `ACTIONID` distinti | è chiave |
|---|---|---|---|
| `mooc_actions.tsv` | 411.749 | 411.749 | sì |
| `mooc_action_features.tsv` | 411.749 | 411.749 | sì |
| `mooc_action_labels.tsv` | 411.749 | 396.633 | **no** |

Un `merge(on="ACTIONID")` perderebbe 15.116 righe e ne duplicherebbe altrettante, restituendo
**lo stesso numero di righe di partenza**: un controllo su `len()` non se ne accorgerebbe.

**Alternative scartate.**
- `merge` anche per le etichette → perde e duplica righe in silenzio.
- `drop_duplicates` sugli `ACTIONID` → sceglierebbe arbitrariamente quale copia tenere.

**Verificato.** `ACTIONID` identico fra azioni e feature (`.equals()` → `True`); 15.116 righe
discordanti fra azioni ed etichette.

---

## 2. Controllo di qualità

| controllo | esito |
|---|---|
| valori mancanti | 0 |
| righe duplicate | 0 |
| duplicati su `USERID`+`TARGETID`+`TIMESTAMP` | 54 coppie, di cui 31 identiche anche nelle `FEATURE` |
| `TIMESTAMP` negativi | 0 |
| valori di `LABEL` | solo 0 e 1 |
| arco temporale | 29,77 giorni |
| `FEATURE0..3` | media 0, deviazione standard 1 (già standardizzate) |

**Le 54 coppie restano.** 23 hanno `FEATURE` diverse: sono due gesti distinti caduti nello stesso
secondo (i `TIMESTAMP` hanno risoluzione di un secondo). Le altre 31 sono indistinguibili, cambia
solo `ACTIONID`. Il dataset non dichiara alcuna de-duplicazione, quindi scartarne una a caso
sarebbe una scelta arbitraria travestita da pulizia. L'effetto è comunque nullo: 62 righe su
411.749.

**Sullo scaling.** Le `FEATURE` sono già standardizzate, qui non serve nulla. Lo scaling del Task 5
si stima dentro una `Pipeline`, mai sui file salvati.

---

## 3. L'unità di analisi è lo studente

**Cosa.** Da 411.749 azioni a 7.047 studenti, con `ABBANDONO = groupby("USERID")["LABEL"].max()`.

**Perché.** `LABEL = 1` marca l'**ultima azione** di chi abbandona: descrive lo studente, non
l'azione. A livello azione la domanda diventerebbe «questa riga è l'ultima del suo gruppo?», già
risolta dall'ordinamento e priva di interesse.

**Verificato.**
- tutte le 4.066 azioni con `LABEL = 1` sono l'ultima del proprio studente;
- nessuno studente ha più di un'azione etichettata;
- 7.047 studenti, 4.066 positivi.

**Conseguenza.** La prevalenza passa dallo **0,99% al 57,7%**: cambia il denominatore
(411.749 → 7.047), il numeratore resta 4.066. Le classi sono quindi bilanciate e nei task
successivi **non serve alcun riequilibrio** — né SMOTE, né pesi di classe. Lo sbilanciamento
estremo era un effetto dell'unità di analisi sbagliata.

---

## 4. Esclusione dell'ultima azione di ogni studente

**Cosa.** Ordinate le azioni per `USERID, TIMESTAMP, ACTIONID` e tolta l'ultima di **ogni**
studente: restano 404.702 azioni.

**Perché.** L'ultima azione è quella su cui l'etichetta è definita: usarla come ingresso
significherebbe descrivere lo studente con il gesto che ne ha determinato la classe.

**Perché per tutti e non solo per i positivi.** Altrimenti la regola dipenderebbe dal target: i
positivi descritti da *n−1* azioni, i negativi da *n*. Applicata a tutti, è cieca rispetto alla
classe.

**Verificato.** 411.749 − 7.047 = 404.702 azioni; tutti i 7.047 studenti sopravvivono, il più
povero con 4 azioni.

---

## 5. Le otto feature

| feature | definizione |
|---|---|
| `n_azioni` | numero di azioni |
| `n_attivita_distinte` | `TARGETID` distinti |
| `n_giorni_attivi` | giorni di calendario distinti (`TIMESTAMP // 86400`) |
| `durata_giorni` | (`TIMESTAMP` max − min) / 86400 |
| `feature0_media` … `feature3_media` | media di `FEATURE0..3` |
| `ABBANDONO` | `max(LABEL)` — **target** |

Valori arrotondati a 3 decimali.

**Come sono calcolate.** Un solo `groupby("USERID")` sulle 404.702 righe della storia, e otto
aggregazioni su quel raggruppamento. Le quattro medie escono da **una** chiamata su tutte e quattro
le colonne insieme (`gruppi[COLONNE_FEATURE].mean()`), non da un giro di ciclo per colonna: pandas
attraversa i dati una volta sola, in C. Vale identico in `preprocessing.py`, che è lo stesso codice
impacchettato in una funzione.

**Perché queste.** Le prime quattro rispondono alle domande naturali su uno studente — quanto,
quanto a lungo, quanto spesso, quanto in ampiezza — e restano leggibili in un grafico e in un
albero. Le ultime quattro sono l'unica informazione descrittiva dell'azione presente nel dataset.

**Alternative scartate.**
- `intervallo_medio_ore` → è esattamente `durata_giorni / (n_azioni − 1)`, quindi ridondante.
- `intervallo_mediano_ore` → mediana 22 secondi (quartili 12,5 e 51,5), da sola dà AUC 0,50.
- somma e massimo di `FEATURE0..3` → raddoppiano le colonne restando ridondanti (la somma correla
  con la media fra 0,66 e 0,91).

**`USERID` non è una feature**, è un identificativo: resta come indice. I file vanno letti con
`index_col="USERID"`.

**Verificato.** 7.047 × 9, tutte numeriche, 0 mancanti, 0 infiniti. Le 256 righe duplicate sono
studenti diversi con lo stesso profilo (pochissime azioni in un giorno solo): sono studenti reali,
restano.

---

## 6. `manuale.csv` — 12 campioni

**Cosa.** 6 studenti che hanno abbandonato e 6 che hanno proseguito, estratti a caso con
`random_state=42`.

**Perché 12 e bilanciati.** La consegna chiede 10–15. Il bilanciamento porta l'entropia iniziale a
1 bit esatto, il che rende leggibili i calcoli del Task 2.

**Perché a caso e non scelti.** Scegliere «i casi più chiari» produrrebbe un file su cui qualunque
classificatore funziona, rendendo la valutazione priva di significato.

---

## 7. `training.csv` — 7.035 campioni

**Cosa.** Tutti gli studenti tranne i 12 di `manuale.csv`.

**Perché tutti.** 7.035 righe con 8 feature sono già poche per un modello: ridurle ulteriormente
non porterebbe alcun vantaggio.

**Perché i 12 sono esclusi.** Saranno già stati usati per costruire i classificatori manuali del
Task 2: lasciarli anche qui significherebbe valutare in parte sugli stessi dati.

**Verificato.** 7.035 righe; prevalenza 57,71%, identica a quella del dataset completo.

---

## 8. Verifica di sanità e nota critica

**Cosa.** Una regressione logistica di default, 5 pieghe, sulle 8 feature.

**Verificato.** ROC-AUC 0,877 — il segnale c'è e il preprocessing non lo ha distrutto.

**Nota critica, da dichiarare all'orale.** La regola «poche azioni → abbandono» è in parte
**tautologica**: un abbandono è per definizione la fine dell'attività, quindi chi abbandona ha
inevitabilmente fatto meno cose. Il modello registra un fatto quasi contabile, non spiega *perché*
uno studente abbandoni.

Verificato in appendice al notebook, ripetendo il calcolo su una **finestra iniziale** di
osservazione: sulle prime 4 azioni di ogni studente l'AUC crolla a **0,535**, sulle prime 10 a
0,542. La predizione ha valore pratico solo se applicata a una finestra iniziale, non all'intera
storia.

---

## 9. `preprocessing.py`, il preprocessing come modulo

**Cosa.** La trasformazione impacchettata in due funzioni: `costruisci_studenti` (log di azioni →
tabella studente) e `carica_per_predire` (legge un file e restituisce `X, y`).

**Perché.** Le 8 feature sono definite da noi e non esistono nel dataset originale: qualunque
tabella su cui si voglia usare il modello va costruita con la **stessa** trasformazione
dell'addestramento. Rifacendola fuori dal modulo, ogni minima differenza cambierebbe i risultati
senza che ce ne accorgiamo. Il modulo garantisce che sia la stessa.

`costruisci_studenti` accetta tre formati:

| formato ricevuto | comportamento |
|---|---|
| log di azioni con `LABEL` | costruisce le feature e il target |
| log di azioni senza `LABEL` | costruisce le feature, `y = None` |
| tabella già a livello studente | la restituisce così com'è |

**Alternativa scartata.** Tenere le colonne originali e classificare a livello azione: la
regressione logistica dà **F1 = 0,000** (average precision 0,031 contro 0,0099 di una scelta
casuale). Il modello impara a rispondere sempre 0 perché i positivi sono lo 0,99%.

**Limite dichiarato.** `n_azioni` e `durata_giorni` dipendono dall'ampiezza della finestra di
osservazione (29,77 giorni). Storie osservate per un periodo diverso produrrebbero feature su una
scala diversa e non confrontabile.

**Verificato.** Il modulo riproduce la tabella del notebook e `training.csv` con `.equals()` →
`True`.

---

## Riepilogo

| passo | risultato |
|---|---|
| unione dei tre file | 411.749 × 9, per posizione |
| controllo qualità | 0 mancanti, 0 duplicati, 0 valori fuori range |
| cambio di unità di analisi | da 411.749 azioni a 7.047 studenti |
| target | `ABBANDONO`, 57,7% di positivi (era 0,99% a livello azione) |
| feature | 8 per studente, escludendo l'ultima azione di ognuno |
| `manuale.csv` / `training.csv` | 12 (6+6) / 7.035 |
| `preprocessing.py` | la stessa trasformazione, richiamabile come funzione |

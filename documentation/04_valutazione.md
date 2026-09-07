# Task 4 — Valutazione e ottimizzazione dei classificatori manuali

**Obiettivo.** Valutare su `training.csv` i due classificatori costruiti a mano nel Task 2 e cercare
di ottimizzarne le prestazioni.

**Prodotti.** `04_valutazione.ipynb` (10 celle di codice). Un solo notebook per entrambi i
classificatori: il protocollo è lo stesso e il punto del task è il **confronto**, quindi duplicarlo
aggiungerebbe pagine senza aggiungere contenuto.

**Risultato.** L'ottimizzazione porta l'accuratezza da **0,7734 a 0,7933** (+2,0 punti, vinti in
tutte e cinque le pieghe). Tutte le configurazioni provate restano però in una fascia di due punti:
il tetto di questi classificatori sui nostri dati è intorno al **79%**.

---

## 1. Il protocollo

**Cosa abbiamo fatto.** Diviso `training.csv` in due parti stratificate e usato in aggiunta la
cross-validation:

| fase | dati | a cosa serve |
|---|---|---|
| ricerca | 70% (4.924 studenti) | provare le configurazioni |
| verifica | 30% (2.111 studenti) | controllare la scelta, una volta sola |
| confronto finale | 5-fold CV su tutto `training.csv` | stimare le prestazioni **con la loro variabilità** |

**Perché non basta una misura sola.** Ottimizzare significa provare molte configurazioni e tenere
la migliore. Se le si prova e le si misura sugli stessi dati, il punteggio finale non è una stima
delle prestazioni ma il **massimo di un insieme di rumore**: si sceglie la configurazione più
fortunata, non la migliore.

**Perché la cross-validation.** Serve soprattutto per la **deviazione standard**. Senza quella non
c'è modo di dire se una differenza di mezzo punto sia un miglioramento o una fluttuazione. Sui
nostri dati la deviazione tipica fra le pieghe è **0,0095**: è il metro con cui va giudicato ogni
risultato di questo task.

---

## 2. Punto di partenza

**Cosa abbiamo fatto.** Riscritte nel notebook le stesse funzioni del Task 2 e verificato che
riproducano esattamente i risultati di allora prima di usarle.

**Cosa abbiamo verificato.** L'albero ritrova `n_attivita_distinte <= 22,5` e fa 1,0000 su
`manuale.csv`; il Naive Bayes fa 1,0000. Sono gli stessi classificatori, non una riscrittura
approssimativa.

**Attenzione a un dettaglio che spiega una differenza fra documenti.** Nel Task 2.1 il Naive Bayes
otteneva **0,7790** su `training.csv`, qui ottiene **0,7902**. Non è una discrepanza: sono due
protocolli diversi. Nel Task 2 *tutto* il modello — priori e probabilità condizionate — era stimato
sui dodici campioni; qui dai dodici campioni vengono solo le **soglie di discretizzazione**, mentre
le condizionate sono ristimate sulle pieghe di addestramento, che è il modo corretto di valutarlo in
cross-validation. L'albero invece dà esattamente lo stesso 0,7734 in entrambi i documenti, perché la
sua regola è completamente fissata dalla soglia e non c'è nulla da ristimare.

| classificatore del Task 2 | accuratezza (CV) | dev.std | F1 |
|---|---|---|---|
| Naive Bayes (soglie di `manuale.csv`) | 0,7902 | 0,0075 | 0,8184 |
| albero (soglia 22,5 dai 12 campioni) | 0,7734 | 0,0088 | 0,8031 |

---

## 3. Ottimizzazione dell'albero

### 3.1 La soglia

**Cosa abbiamo verificato.** Sulla parte di ricerca la soglia migliore è **28**, non 22,5. Sul 30%
di verifica il guadagno si conferma, ma è piccolo:

| soglia | stimata su | accuratezza in verifica | F1 |
|---|---|---|---|
| 22,5 | 12 campioni (Task 2) | 0,7603 | 0,7936 |
| 28 | 4.924 studenti | **0,7740** | 0,8167 |

### 3.2 La feature

**Cosa abbiamo fatto.** Provata ogni feature come nodo singolo, in **entrambe le direzioni**
(`<=` e `>`), perché per alcune l'abbandono sta sopra la soglia e non sotto.

**Cosa abbiamo verificato.**

| feature | regola migliore | acc. ricerca | acc. verifica |
|---|---|---|---|
| `n_azioni` | `<= 55` | **0,7912** | 0,7693 |
| `n_attivita_distinte` | `<= 28` | 0,7837 | **0,7740** |
| `n_giorni_attivi` | `<= 4` | 0,7807 | 0,7518 |
| `durata_giorni` | `<= 14,193` | 0,7581 | 0,7513 |
| `feature0_media` | `<= −0,160` | 0,7053 | 0,6921 |
| `feature3_media` | `<= −0,067` | 0,6363 | 0,6324 |
| `feature2_media` | `> −0,023` | 0,6054 | 0,5973 |
| `feature1_media` | `<= 0,518` | 0,5674 | 0,5694 |

**Il dettaglio che vale la pena raccontare.** La feature migliore *in ricerca* è `n_azioni`, ma
sulla verifica le prime due **si invertono**: chi vince la ricerca perde il controllo. È
esattamente il motivo per cui le due misure vanno tenute separate — e la differenza resta comunque
dentro il rumore.

Si conferma anche il quadro del Task 3: le tre feature inerti stanno fra 0,56 e 0,61, cioè poco
sopra il caso.

### 3.3 La profondità

**Cosa abbiamo fatto.** L'albero del Task 2 aveva un solo nodo perché i dodici campioni erano
separabili. Sui dati veri lo facciamo crescere, misurando in cross-validation.

**Cosa abbiamo verificato.**

| max_depth | accuratezza | dev.std | F1 |
|---|---|---|---|
| 1 | 0,7831 | 0,0086 | 0,8162 |
| 2 | 0,7831 | 0,0086 | 0,8162 |
| 3 | 0,7842 | 0,0097 | 0,8177 |
| 4 | 0,7893 | 0,0095 | 0,8235 |
| 5 | **0,7906** | **0,0141** | 0,8228 |

**Il guadagno c'è ma metà dell'albero non serve.** Da profondità 1 a 5 si guadagnano meno di
0,8 punti, mentre la deviazione standard **cresce di oltre il 60%** (0,0086 → 0,0141): il modello diventa
più instabile fra una piega e l'altra, che è il segnale classico di sovradattamento.

Il conteggio dei nodi spiega perché. Nell'albero di profondità 5 ci sono 29 nodi di decisione, e in
**16 di essi (55%) entrambi i rami portano alla stessa classe**: riducono un po' di entropia ma non
cambiano *nessuna* predizione. È anche il motivo per cui profondità 1 e 2 danno risultati identici.

Il primo livello sceglie `n_azioni <= 50` con IG 0,2451; già il secondo scende a IG 0,0456.

---

## 4. Ottimizzazione del Naive Bayes

### 4.1 Il numero di intervalli

**Cosa abbiamo verificato.** Suddividere ogni feature in più di due intervalli (quantili stimati
sull'addestramento di ogni piega) aiuta poco e smette presto:

| intervalli | accuratezza | dev.std |
|---|---|---|
| 2 | 0,7750 | 0,0082 |
| 3 | 0,7811 | 0,0113 |
| 4 | **0,7822** | 0,0088 |
| 5 | 0,7819 | 0,0097 |
| 8 | 0,7790 | 0,0113 |

### 4.2 Da dove conviene prendere le soglie

**L'osservazione di partenza.** Il Naive Bayes del Task 2 usa le mediane di **dodici** campioni e
ottiene 0,7902; le versioni che stimano i quantili su **migliaia** di studenti fanno peggio. È
controintuitivo e merita una spiegazione, non un'alzata di spalle.

**L'ipotesi.** `manuale.csv` è bilanciato per costruzione (6 e 6), mentre `training.csv` ha il 57,7%
di positivi. La mediana di una popolazione sbilanciata è spostata verso la classe maggioritaria e
quindi taglia in un punto peggiore. Se l'ipotesi è giusta, **ribilanciare `training.csv` prima di
calcolare le mediane deve recuperare il divario**.

**Cosa abbiamo verificato.**

| origine delle soglie | accuratezza | dev.std | confronto appaiato con `manuale.csv` |
|---|---|---|---|
| punto medio fra le medie di classe | **0,7933** | 0,0080 | −0,0031 (manuale vince 2/5) |
| mediane di `manuale.csv` (Task 2) | 0,7902 | 0,0075 | — |
| mediane di `training.csv` ribilanciato | 0,7889 | 0,0076 | +0,0013 (manuale vince 2/5) |
| mediane di `training.csv` | 0,7775 | 0,0086 | **+0,0127 (manuale vince 5/5)** |

**Ipotesi confermata.** Le mediane di `training.csv` così com'è perdono **tutte e cinque** le pieghe
contro quelle dei dodici campioni. Appena si ribilancia il dataset prima di calcolarle, il divario
sparisce; e prendere il punto medio fra le medie delle due classi fa addirittura leggermente meglio.

Il Naive Bayes del Task 2 non era quindi fortunato: era **tarato su un campione bilanciato**. Non è
la numerosità a determinare la bontà di una soglia di taglio, ma il fatto che il campione da cui la
si stima rappresenti equamente le due classi.

---

## 5. Confronto finale

| configurazione | accuratezza (CV) | dev.std | scarto dal migliore |
|---|---|---|---|
| Naive Bayes, soglie dalle medie di classe | **0,7933** | 0,0080 | — |
| albero ricresciuto sui dati, profondità 5 | 0,7906 | 0,0141 | 0,0027 |
| Naive Bayes del Task 2 | 0,7902 | 0,0075 | 0,0031 |
| albero ricresciuto sui dati, profondità 4 | 0,7893 | 0,0095 | 0,0040 |
| Naive Bayes, 4 intervalli sui quantili | 0,7822 | 0,0088 | 0,0111 |
| albero del Task 2 | 0,7734 | 0,0088 | 0,0199 |

Baseline «rispondi sempre abbandono»: 0,5771.

---

## 6. Analisi critica

**L'ottimizzazione funziona, ma il margine è sottile.** Dal punto di partenza del Task 2 alla
configurazione migliore si guadagnano 2,0 punti, vinti in tutte e cinque le pieghe. Con una
deviazione standard tipica di 0,0095, il guadagno vale però poco più di due deviazioni standard: è
un miglioramento **reale ma modesto**, e va dichiarato come tale.

**Il tetto è intorno al 79%.** Tutte le configurazioni provate — soglia diversa, feature diversa,
profondità da 1 a 5, da 2 a 8 intervalli, quattro modi di stimare le soglie — cadono in una fascia
di circa due punti. Non è un limite dell'ottimizzazione ma delle **feature**: come mostrato nel
Task 3, tre delle otto sono rumore e le altre cinque misurano in larga parte la stessa cosa.
Nessuna regolazione di un classificatore semplice può estrarre informazione che nei dati non c'è.

**I due modelli restano equivalenti.** Le prime quattro righe della tabella finale stanno in 0,4
punti, cioè meno di mezza deviazione standard. Non c'è un vincitore fra albero e Naive Bayes, ed è
coerente con la spiegazione data nel Task 2: l'albero usa una feature, il Naive Bayes ne usa otto,
ma quelle otto ripetono in gran parte la stessa informazione.

**La lezione metodologica.** Il caso delle soglie del Naive Bayes è il risultato più utile del task:
la configurazione che sembrava fortunata era in realtà corretta per una ragione precisa — il
bilanciamento — e ce ne siamo accorti solo formulando un'ipotesi e provandola con un esperimento
controllato, non guardando la classifica dei punteggi.

**Cosa portiamo al Task 5.** Serve più **capacità di modello**, non più regolazione fine: frontiere
non lineari e insiemi di alberi. Il protocollo invece resta questo — ricerca e verifica separate,
cross-validation per la variabilità — perché è l'unico modo di distinguere un miglioramento da una
fluttuazione.

---

## Riepilogo

| passo | risultato |
|---|---|
| protocollo | 70/30 stratificato per la ricerca, 5-fold CV per il confronto |
| controprova sulle funzioni | riproducono il Task 2 esattamente (soglia 22,5, accuratezza 1,0000) |
| albero, soglia ottimizzata | 22,5 → 28 |
| albero, profondità | guadagno < 0,8 punti, dev.std da 0,0086 a 0,0141 |
| albero, nodi inutili | 16 su 29 (55%) non cambiano alcuna predizione |
| Naive Bayes, intervalli | 4 è il massimo utile (0,7822) |
| Naive Bayes, soglie | contano il bilanciamento del campione, non la sua numerosità |
| migliore configurazione | Naive Bayes con soglie dalle medie di classe, 0,7933 |
| guadagno complessivo | +2,0 punti (0,7734 → 0,7933), circa 2 deviazioni standard |

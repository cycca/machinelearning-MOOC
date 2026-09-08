# Task 4 — Valutazione e ottimizzazione dei classificatori manuali

**Obiettivo.** Valutare su `training.csv` i due classificatori manuali del Task 2 e cercare
di ottimizzarne le prestazioni.

**Risultato.** L'ottimizzazione porta l'accuratezza da **0,7734 a 0,7933** (+2,0 punti, vinti in
tutte e cinque le pieghe). Tutte le configurazioni provate restano però in una fascia di due punti:
il tetto di questi classificatori è intorno al **79%**.

Un solo notebook per entrambi i classificatori: il protocollo è lo stesso e il punto del task è il
**confronto**.

---

## 1. Il protocollo

| fase | dati | a cosa serve |
|---|---|---|
| ricerca | 70% (4.924 studenti) | provare le configurazioni |
| verifica | 30% (2.111 studenti) | controllare la scelta, una volta sola |
| confronto finale | 5-fold CV su tutto `training.csv` | stimare le prestazioni **con la variabilità** |

**Perché non basta una misura sola.** Ottimizzare significa provare molte configurazioni e tenere la
migliore. Misurandole sugli stessi dati su cui le si prova, il punteggio finale è il **massimo di un
insieme di rumore**: si sceglie la configurazione più fortunata, non la migliore.

**Perché la cross-validation.** Serve per la **deviazione standard**, che sui nostri dati vale
**0,0095**. È il metro con cui va giudicato ogni risultato del task: una differenza inferiore a un
punto non è distinguibile dal rumore.

---

## 2. Punto di partenza

**Nota sull'implementazione.** Le funzioni del Task 2 sono riscritte qui in forma vettorizzata:
la ricerca dello split valuta tutte le ~320 coppie (feature, soglia) del nodo in un'unica matrice
`campioni × coppie`, i conteggi del Naive Bayes escono da un solo `np.bincount`, e la predizione
scende l'albero a maschere invece che riga per riga. Serve, perché questo notebook fa crescere 25
alberi: **67,7 secondi con i cicli, 6,6 vettorizzato**, con output identici al carattere. Dettagli
in «Come è scritto il codice» di [relazione.md](relazione.md).

**Verificato.** Le funzioni riscritte nel notebook riproducono il Task 2 esattamente: il decision
tree ritrova `n_attivita_distinte <= 22,5` e fa 1,0000 su `manuale.csv`, il Naive Bayes fa 1,0000.

| classificatore del Task 2 | accuratezza (CV) | dev.std | F1 |
|---|---|---|---|
| Naive Bayes (soglie di `manuale.csv`) | 0,7902 | 0,0075 | 0,8184 |
| decision tree (soglia 22,5) | 0,7734 | 0,0088 | 0,8031 |

**Un dettaglio che spiega una differenza fra documenti.** Nel Task 2.1 il Naive Bayes otteneva
**0,7790**, qui ottiene **0,7902**. Sono due protocolli diversi: nel Task 2 *tutto* il modello —
priori e condizionate — era stimato sui dodici campioni; qui dai dodici campioni vengono solo le
**soglie di discretizzazione**, mentre le condizionate sono ristimate sulle pieghe di addestramento,
che è il modo corretto di valutarlo in cross-validation. Il decision tree invece dà lo stesso 0,7734
in entrambi i documenti, perché la sua regola è completamente fissata dalla soglia.

---

## 3. Ottimizzazione del decision tree

### 3.1 La soglia

| soglia | stimata su | accuratezza in verifica | F1 |
|---|---|---|---|
| 22,5 | 12 campioni (Task 2) | 0,7603 | 0,7936 |
| 28 | 4.924 studenti | **0,7740** | 0,8167 |

### 3.2 La feature

Provata ogni feature come nodo singolo, in **entrambe le direzioni** (`<=` e `>`), perché per alcune
l'abbandono sta sopra la soglia.

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

La feature migliore *in ricerca* è `n_azioni`, ma sulla verifica le
prime due **si invertono**: chi vince la ricerca perde il controllo. È il motivo per cui le due
misure vanno tenute separate. Si conferma anche il quadro del Task 3: le tre feature inerti stanno
fra 0,56 e 0,61.

### 3.3 La profondità

| max_depth | accuratezza | dev.std | F1 |
|---|---|---|---|
| 1 | 0,7831 | 0,0086 | 0,8162 |
| 2 | 0,7831 | 0,0086 | 0,8162 |
| 3 | 0,7842 | 0,0097 | 0,8177 |
| 4 | 0,7893 | 0,0095 | 0,8235 |
| 5 | **0,7906** | **0,0141** | 0,8228 |

**Il guadagno c'è ma metà dell'albero non serve.** Da profondità 1 a 5 si guadagnano meno di 0,8
punti, mentre la deviazione standard cresce di oltre il 60% (0,0086 → 0,0141): overfitting.

Il conteggio dei nodi spiega perché: nell'albero di profondità 5 ci sono 29 nodi di decisione, e in
**16 di essi (55%) entrambi i rami portano alla stessa classe**. Riducono entropia ma non cambiano
*nessuna* predizione — ed è anche il motivo per cui profondità 1 e 2 danno risultati identici.

---

## 4. Ottimizzazione del Naive Bayes

### 4.1 Il numero di intervalli

| intervalli | accuratezza | dev.std |
|---|---|---|
| 2 | 0,7750 | 0,0082 |
| 3 | 0,7811 | 0,0113 |
| 4 | **0,7822** | 0,0088 |
| 5 | 0,7819 | 0,0097 |
| 8 | 0,7790 | 0,0113 |

### 4.2 Da dove conviene prendere le soglie

**L'osservazione di partenza.** Il Naive Bayes tarato su **dodici** campioni ottiene 0,7902; le
versioni che stimano i quantili su **migliaia** di studenti fanno peggio. È controintuitivo e merita
una spiegazione, non un'alzata di spalle.

**L'ipotesi.** `manuale.csv` è bilanciato per costruzione (6 e 6), `training.csv` ha il 57,7% di
positivi. La mediana di una popolazione sbilanciata è spostata verso la classe maggioritaria e
taglia in un punto peggiore. Se l'ipotesi è giusta, **ribilanciare `training.csv` deve recuperare il
divario**.

| origine delle soglie | accuratezza | dev.std | confronto appaiato |
|---|---|---|---|
| punto medio fra le medie di classe | **0,7933** | 0,0080 | −0,0031 (manuale vince 2/5) |
| mediane di `manuale.csv` (Task 2) | 0,7902 | 0,0075 | — |
| mediane di `training.csv` ribilanciato | 0,7889 | 0,0076 | +0,0013 (manuale vince 2/5) |
| mediane di `training.csv` | 0,7775 | 0,0086 | **+0,0127 (manuale vince 5/5)** |

**Ipotesi confermata.** Le mediane di `training.csv` così com'è perdono **tutte e cinque** le pieghe.
Appena si ribilancia il dataset il divario sparisce. Il Naive Bayes del Task 2 non era fortunato: era
**tarato su un campione bilanciato**. Per una soglia di taglio non conta la numerosità del campione,
ma che rappresenti equamente le due classi.

---

## 5. Confronto finale

| configurazione | accuratezza (CV) | dev.std | scarto dal migliore |
|---|---|---|---|
| Naive Bayes, soglie dalle medie di classe | **0,7933** | 0,0080 | — |
| decision tree ricresciuto, profondità 5 | 0,7906 | 0,0141 | 0,0027 |
| Naive Bayes del Task 2 | 0,7902 | 0,0075 | 0,0031 |
| decision tree ricresciuto, profondità 4 | 0,7893 | 0,0095 | 0,0040 |
| Naive Bayes, 4 intervalli sui quantili | 0,7822 | 0,0088 | 0,0111 |
| decision tree del Task 2 | 0,7734 | 0,0088 | 0,0199 |

Baseline «rispondi sempre abbandono»: 0,5771.

---

## 6. Analisi critica

**Il margine è sottile.** 2,0 punti guadagnati, vinti in tutte e cinque le pieghe, ma con una
deviazione standard tipica di 0,0095 valgono poco più di due deviazioni standard: un miglioramento
**reale ma modesto**, e va dichiarato come tale.

**Il tetto è intorno al 79%.** Tutte le configurazioni provate cadono in una fascia di due punti. Non
è un limite dell'ottimizzazione ma delle **feature**: tre delle otto sono rumore e le altre cinque
misurano in larga parte la stessa cosa.

**I due modelli restano equivalenti.** Le prime quattro righe della tabella finale stanno in 0,4
punti, meno di mezza deviazione standard. Coerente con il Task 2: il decision tree usa una feature,
il Naive Bayes otto, ma quelle otto ripetono la stessa informazione.

**La lezione metodologica.** Il caso delle soglie è il risultato più utile: la configurazione che
sembrava fortunata era corretta per una ragione precisa — il bilanciamento — e ce ne siamo accorti
formulando un'ipotesi e provandola, non guardando la classifica dei punteggi.

**Cosa portiamo al Task 5.** Serve più **capacità di modello**, non più regolazione fine. Il
protocollo invece resta questo.

---

## Riepilogo

| passo | risultato |
|---|---|
| protocollo | 70/30 per la ricerca, 5-fold CV per il confronto |
| controprova sulle funzioni | riproducono il Task 2 esattamente |
| decision tree, soglia | 22,5 → 28 |
| decision tree, profondità | guadagno < 0,8 punti, dev.std da 0,0086 a 0,0141 |
| decision tree, nodi inutili | 16 su 29 (55%) |
| Naive Bayes, intervalli | 4 è il massimo utile (0,7822) |
| Naive Bayes, soglie | conta il bilanciamento del campione, non la numerosità |
| migliore configurazione | Naive Bayes con soglie dalle medie di classe, 0,7933 |
| guadagno complessivo | +2,0 punti, circa 2 deviazioni standard |

# Task 3 — Data quality e analisi esplorativa

**Obiettivo.** Verificare che `training.csv` non contenga osservazioni palesemente errate ed
effettuare l'analisi esplorativa, con rappresentazione grafica dei risultati.

**Prodotti.** `03_analisi_esplorativa.ipynb` (11 celle di codice, 6 figure).

**Risultato.** Il dataset **non richiede pulizia**: zero valori mancanti, nessun valore fuori range,
nessuna riga da rimuovere. L'analisi individua però tre proprietà che condizionano i Task 4 e 5:
forte collinearità fra le feature di quantità, tre feature su otto prive di segnale, e distribuzioni
molto asimmetriche.

---

## 1. Struttura e statistiche descrittive

**Cosa abbiamo fatto.** Tabella dei momenti principali con l'aggiunta dell'indice di asimmetria.

**Cosa abbiamo verificato.** 7.035 righe × 9 colonne, tutte numeriche. Nessun valore impossibile: i
conteggi partono da 1 e `durata_giorni` sta fra 0 e 29,135, dentro i 29,77 giorni della finestra di
osservazione del dataset.

Due gruppi di feature si comportano in modo molto diverso:

| gruppo | asimmetria | lettura |
|---|---|---|
| `n_azioni`, `n_attivita_distinte`, `n_giorni_attivi`, `durata_giorni` | 0,32 – 1,64 | code a destra: molti studenti con storie brevi, pochi con storie lunghe |
| `feature2_media`, `feature3_media` | **54,8** e **16,1** | code estreme: massimo 20,5 contro terzo quartile 0,07 |

---

## 2. Controlli di qualità

**Cosa abbiamo fatto.** Oltre ai controlli standard (mancanti, duplicati, range) abbiamo verificato
**cinque vincoli logici che devono valere per costruzione**: se uno di questi fallisse
indicherebbe un errore nel preprocessing del Task 1.

**Cosa abbiamo verificato.** Zero valori mancanti, `USERID` univoco, tutti i vincoli soddisfatti
tranne uno:

| vincolo | esito |
|---|---|
| `n_attivita_distinte <= n_azioni` | OK |
| `n_giorni_attivi <= n_azioni` | OK |
| `n_azioni >= 1` | OK |
| `durata_giorni >= 0` | OK |
| `durata_giorni <= 29,77` (finestra osservata) | OK |
| `n_giorni_attivi <= durata_giorni + 1` | **falso, in 392 righe (5,6%)** |

### 2.1 Il vincolo che fallisce non è un errore

**Perché.** `n_giorni_attivi` conta i **giorni di calendario distinti** (il bucket
`TIMESTAMP // 86400`), mentre `durata_giorni` misura la distanza fra prima e ultima azione. Due
azioni a cavallo della mezzanotte cadono in bucket diversi pur distando pochi minuti.

**Cosa abbiamo verificato.** Le violazioni riguardano tutte studenti con durata brevissima, e il
conto torna sui timestamp grezzi. Lo studente 9 agisce ai secondi 38.414 e 122.038–122.059: due
bucket giornalieri (0 e 1), ma `(max − min)/86400 = 0,968` giorni. Quindi 2 > 1,968.

**Conseguenza da dichiarare.** La feature resta valida, ma va interpretata correttamente:
`n_giorni_attivi` misura **in quanti giorni di calendario lo studente è comparso**, non per quanti
giorni è rimasto attivo.

### 2.2 Righe duplicate: non vanno rimosse

**Cosa abbiamo verificato.** 358 righe hanno un profilo identico a un'altra (255 ripetizioni in 103
gruppi distinti). `USERID` è però univoco: sono studenti **diversi** con lo stesso profilo.

**Perché non sono errori.** Hanno tutti storie brevissime — mediana 5 azioni contro 37 del dataset,
massimo 9. Due studenti che compiono quattro azioni sulla stessa attività nello stesso giorno
producono inevitabilmente le stesse otto feature: è un limite di risoluzione della
rappresentazione, non un difetto dei dati.

**Perché rimuoverle sarebbe un errore.** Toglierebbe dal dataset proprio gli studenti con storia
breve, che sono in grande maggioranza abbandoni: distorcerebbe il target.

### 2.3 Valori estremi: non vanno rimossi

**Cosa abbiamo verificato.** 6 studenti hanno `feature2_media > 1` e 39 hanno `feature3_media > 1`,
contro un 99° percentile di 0,33 e 0,63. Il massimo è 20,5, dello studente 5103.

**Perché non sono errori.** Sono medie calcolate su decine di azioni reali: lo studente 5103 ha 84
azioni. Non esiste un criterio per dichiararli sbagliati senza inventarsi una soglia.

**Conseguenza per il Task 5.** Li teniamo, ma `StandardScaler` stima media e deviazione standard e
quindi risente di code così pesanti: andrà confrontato con uno scaler robusto.

---

## 3. La variabile target

**Cosa abbiamo verificato.** 4.060 abbandoni (57,7%) contro 2.975 (42,3%): classi **quasi
bilanciate**. È la conferma grafica del risultato del Task 1 — lo sbilanciamento estremo (0,99%)
esisteva solo a livello di azione, cioè nell'unità di analisi sbagliata. Nel Task 5 non servirà
alcun riequilibrio.

---

## 4. Distribuzione delle feature

**Cosa abbiamo fatto.** Un boxplot per classe di ciascuna delle otto feature, ciascuno con la
**propria scala**, più gli istogrammi per classe delle quattro feature di quantità.

**Perché una scala per riquadro e senza outlier.** Le otto feature hanno ordini di grandezza
diversi (da `n_azioni` fino a 504 a `feature3_media` intorno a 0): una scala comune renderebbe
illeggibili sette riquadri su otto. Gli outlier non sono disegnati perché schiaccerebbero i
quartili — sono già stati analizzati numericamente al punto 2.3.

**Cosa abbiamo verificato.** Le quattro feature di quantità separano nettamente i due gruppi, con
gli abbandoni accumulati sui valori bassi. Le `feature*_media` hanno mediane e quartili quasi
sovrapposti, con la parziale eccezione di `feature0_media`.

Le distribuzioni sono inoltre **fortemente asimmetriche e per nulla normali**: `durata_giorni` è
addirittura bimodale, con un picco a zero (chi esaurisce tutto in un istante) e uno oltre i 20
giorni. È una conferma indipendente della scelta fatta nel Task 2.1, dove il Naive Bayes gaussiano
era stato scartato: l'ipotesi di normalità non regge su questi dati.

---

## 5. Perché tre feature su otto sono inerti

**Cosa abbiamo fatto.** Verificato quanto siano concentrate le quattro FEATURE **grezze** sulle
411.749 azioni del dataset originale.

**Cosa abbiamo verificato.** Sono standardizzate (deviazione standard 1,000) ma quasi costanti:

| feature grezza | quota sul valore più frequente | valori distinti |
|---|---|---|
| `FEATURE3` | **93,5%** | 164 |
| `FEATURE0` | **86,9%** | 27 |
| `FEATURE1` | **83,0%** | 16 |
| `FEATURE2` | 64,5% | 87 |

**Perché è il risultato più importante dell'EDA.** Fare la media di una variabile che non varia
produce una feature che non varia. Una sola causa spiega tre osservazioni fatte in task diversi:

| osservazione | dove era emersa |
|---|---|
| `feature1/2/3_media` correlate a zero col target | Task 1 |
| varianza degenere di `feature3_media` (0,00018) che faceva esplodere il NB gaussiano | Task 2.1 |
| un terzo degli studenti con `feature0_media` esattamente al minimo | Task 2.1 |

Il difetto è **del dataset originale, non del nostro preprocessing**: è una distinzione che vale la
pena saper fare all'orale.

---

## 6. Correlazioni

**Cosa abbiamo fatto.** Matrice di correlazione di Pearson su tutte le colonne e pairplot delle
quattro feature più correlate al target.

**Cosa abbiamo verificato.** Due blocchi nettamente distinti:

| feature | correlazione con `ABBANDONO` |
|---|---|
| `n_attivita_distinte` | −0,58 |
| `n_azioni` | −0,56 |
| `n_giorni_attivi` | −0,55 |
| `durata_giorni` | −0,55 |
| `feature0_media` | −0,34 |
| `feature2_media` | +0,08 |
| `feature1_media` | −0,03 |
| `feature3_media` | +0,01 |

Le quattro feature di quantità sono anche **fortemente correlate fra loro**: da 0,69 a **0,91**, con
il massimo fra `n_azioni` e `n_attivita_distinte`.

Il pairplot mostra la stessa cosa in forma geometrica: le nuvole si separano lungo gli assi delle
quantità ma restano sovrapposte lungo `feature0_media`, e il pannello
`n_azioni` × `n_attivita_distinte` è una striscia quasi rettilinea — l'aspetto grafico di una
correlazione di 0,91.

---

## 7. Conclusioni per i task successivi

1. **Nessuna osservazione palesemente errata.** Il dataset non richiede pulizia: l'unico vincolo
   violato si è rivelato una conseguenza corretta della definizione di «giorno di calendario».
2. **Nessuna riga va rimossa.** Duplicati e valori estremi sono entrambi legittimi, e rimuoverli
   distorcerebbe il target.
3. **Collinearità fino a 0,91** fra le feature di quantità: i coefficienti di un modello lineare non
   saranno interpretabili singolarmente, e l'importanza delle feature va letta con cautela. Spiega
   anche il risultato del Task 2: albero (una feature) e Naive Bayes (otto) ottengono quasi lo
   stesso punteggio, perché le sette in più ripetono la stessa informazione.
4. **Tre feature su otto sono rumore**, per un difetto del dataset originale. Le teniamo — scartarle
   guardando questi dati sarebbe una scelta presa sul test — ma non ci si aspetta nulla da loro.
5. **Distribuzioni asimmetriche e code pesanti**: conviene confrontare `StandardScaler` con uno
   scaler robusto, e i modelli ad albero, insensibili alle trasformazioni monotone, partono
   avvantaggiati.

---

## Riepilogo

| controllo | esito |
|---|---|
| valori mancanti | 0 |
| `USERID` duplicati | 0 |
| vincoli logici | 5 su 6 soddisfatti; il sesto spiegato e non correttivo |
| righe con profilo identico | 358 (255 ripetizioni), tutte legittime, nessuna rimossa |
| valori estremi | 6 + 39 studenti, tutti legittimi, nessuno rimosso |
| classi | 57,7% / 42,3%, nessun riequilibrio necessario |
| feature informative | 5 su 8 (le quattro quantità più `feature0_media`) |
| collinearità massima | 0,91 fra `n_azioni` e `n_attivita_distinte` |

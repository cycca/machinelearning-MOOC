# Task 3 — Data quality e analisi esplorativa

**Obiettivo.** Verificare che `training.csv` non contenga osservazioni palesemente errate ed
effettuare l'analisi esplorativa, con rappresentazione grafica.

**Risultato.** Il dataset **non richiede pulizia**: zero mancanti, nessun valore fuori range, nessuna
riga da rimuovere. Emergono però tre proprietà che condizionano i Task 4 e 5: collinearità fino a
0,91, tre feature su otto prive di segnale, distribuzioni molto asimmetriche.

---

## 1. Statistiche descrittive

7.035 righe × 9 colonne, tutte numeriche. Nessun valore impossibile: i conteggi partono da 1 e
`durata_giorni` sta fra 0 e 29,135, dentro i 29,77 giorni di osservazione.

| gruppo | asimmetria | lettura |
|---|---|---|
| `n_azioni`, `n_attivita_distinte`, `n_giorni_attivi`, `durata_giorni` | 0,32 – 1,64 | code a destra: molti studenti con storie brevi |
| `feature2_media`, `feature3_media` | **54,8** e **16,1** | code estreme: massimo 20,5 contro terzo quartile 0,07 |

---

## 2. Controlli di qualità

**Cosa.** Oltre ai controlli standard abbiamo verificato cinque **vincoli logici che devono valere
per costruzione**: se uno fallisse, indicherebbe un errore nel preprocessing del Task 1.

| vincolo | esito |
|---|---|
| `n_attivita_distinte <= n_azioni` | OK |
| `n_giorni_attivi <= n_azioni` | OK |
| `n_azioni >= 1` | OK |
| `durata_giorni >= 0` | OK |
| `durata_giorni <= 29,77` | OK |
| `n_giorni_attivi <= durata_giorni + 1` | **falso, in 392 righe (5,6%)** |

Zero valori mancanti, `USERID` univoco.

### 2.1 Il vincolo che fallisce non è un errore

`n_giorni_attivi` conta i **giorni di calendario distinti** (`TIMESTAMP // 86400`), mentre
`durata_giorni` misura la distanza fra prima e ultima azione. Due azioni a cavallo della mezzanotte
cadono in bucket diversi pur distando pochi minuti.

**Verificato sui timestamp grezzi.** Lo studente 9 agisce ai secondi 38.414 e 122.038–122.059: due
bucket giornalieri, ma `(max − min)/86400 = 0,968` giorni. Quindi 2 > 1,968.

**Conseguenza da dichiarare.** `n_giorni_attivi` misura **in quanti giorni di calendario lo studente
è comparso**, non per quanti giorni è rimasto attivo.

### 2.2 Righe duplicate: non vanno rimosse

358 righe hanno un profilo identico a un'altra (255 ripetizioni in 103 gruppi), ma `USERID` è
univoco: sono studenti **diversi**, tutti con storie brevissime (mediana 5 azioni contro 37,
massimo 9). Due studenti che compiono quattro azioni sulla stessa attività nello stesso giorno
producono inevitabilmente le stesse otto feature.

Rimuoverle toglierebbe dal dataset proprio gli studenti con storia breve, in grande maggioranza
abbandoni: distorcerebbe il target.

### 2.3 Valori estremi: non vanno rimossi

6 studenti hanno `feature2_media > 1` e 39 hanno `feature3_media > 1`, contro un 99° percentile di
0,33 e 0,63. Il massimo è 20,5, dello studente 5103 — che ha però 84 azioni: sono medie reali, non
refusi.

**Conseguenza per il Task 5.** `StandardScaler` stima media e deviazione standard, quindi risente di
code così pesanti: andrà confrontato con uno scaler resistente agli outlier.

---

## 3. La variabile target

4.060 abbandoni (57,7%) contro 2.975 (42,3%): classi **quasi bilanciate**. È la conferma grafica del
Task 1 — lo sbilanciamento estremo (0,99%) esisteva solo a livello di azione. Nel Task 5 non servirà
alcun riequilibrio.

---

## 4. Distribuzione delle feature

**Cosa.** Un boxplot per classe di ciascuna feature, ciascuno con la **propria scala**, più gli
istogrammi per classe delle quattro feature di quantità.

**Perché una scala per riquadro e senza outlier.** Le feature vanno da `n_azioni` fino a 504 a
`feature3_media` intorno a 0: una scala comune renderebbe illeggibili sette riquadri su otto. Gli
outlier non sono disegnati perché schiaccerebbero i quartili — sono già analizzati al punto 2.3.

![Distribuzione di ogni feature per classe](figure/eda_boxplot_per_classe.png)

**Verificato.** Le quattro feature di quantità separano nettamente i gruppi, con gli abbandoni sui
valori bassi. Le `feature*_media` hanno mediane e quartili quasi sovrapposti, tranne parzialmente
`feature0_media`.

Le distribuzioni sono **fortemente asimmetriche e per nulla normali**: `durata_giorni` è addirittura
bimodale, con un picco a zero e uno oltre i 20 giorni. È una conferma indipendente della scelta del
Task 2.1, dove il Naive Bayes gaussiano era stato scartato.

---

## 5. Perché tre feature su otto sono inerti

**Cosa.** Verificato quanto siano concentrate le quattro FEATURE **grezze** sulle 411.749 azioni
originali.

| feature grezza | quota sul valore più frequente | valori distinti |
|---|---|---|
| `FEATURE3` | **93,5%** | 164 |
| `FEATURE0` | **86,9%** | 27 |
| `FEATURE1` | **83,0%** | 16 |
| `FEATURE2` | 64,5% | 87 |

![Le FEATURE grezze sono quasi costanti](figure/eda_feature_grezze.png)

Sono standardizzate (deviazione standard 1,000) ma quasi costanti. Fare la media di una variabile
che non varia produce una feature che non varia.

**Una sola causa spiega tre osservazioni fatte in task diversi:**

| osservazione | dove era emersa |
|---|---|
| `feature1/2/3_media` correlate a zero col target | Task 1 |
| varianza degenere (0,00018) che faceva esplodere il NB gaussiano | Task 2.1 |
| un terzo degli studenti con `feature0_media` al minimo | Task 2.1 |

**Il difetto è del dataset originale, non del nostro preprocessing.**

---

## 6. Correlazioni

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

![Matrice di correlazione](figure/eda_correlazione.png)

Le quattro feature di quantità sono anche **fortemente correlate fra loro**: da 0,69 a **0,91**, con
il massimo fra `n_azioni` e `n_attivita_distinte`.

Il pairplot lo mostra geometricamente: le nuvole si separano lungo gli assi delle quantità ma non
lungo `feature0_media`, e il pannello `n_azioni` × `n_attivita_distinte` è una striscia quasi
rettilinea.

---

## 7. Conclusioni per i task successivi

1. **Nessuna osservazione palesemente errata**: l'unico vincolo violato è una conseguenza corretta
   della definizione di «giorno di calendario».
2. **Nessuna riga va rimossa**: duplicati e valori estremi sono entrambi legittimi.
3. **Collinearità fino a 0,91**: i coefficienti di un modello lineare non saranno interpretabili
   singolarmente. Spiega anche il risultato del Task 2, dove decision tree (una feature) e Naive
   Bayes (otto) ottengono quasi lo stesso punteggio.
4. **Tre feature su otto sono rumore**, per un difetto del dataset originale. Le teniamo — scartarle
   guardando questi dati sarebbe una scelta presa sul test.
5. **Distribuzioni asimmetriche**: conviene confrontare `StandardScaler` con uno scaler resistente
   agli outlier, e i modelli ad albero partono avvantaggiati.

---

## Riepilogo

| controllo | esito |
|---|---|
| valori mancanti / `USERID` duplicati | 0 / 0 |
| vincoli logici | 5 su 6; il sesto spiegato e non correttivo |
| righe con profilo identico | 358, tutte legittime, nessuna rimossa |
| valori estremi | 45 studenti, tutti legittimi, nessuno rimosso |
| classi | 57,7% / 42,3%, nessun riequilibrio necessario |
| feature informative | 5 su 8 |
| collinearità massima | 0,91 |

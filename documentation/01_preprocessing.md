# Task 1 — Preprocessing e preparazione dei dati

**Obiettivo.** Trasformare *act-mooc* in un formato leggibile da un classificatore ed estrarre
`manuale.csv` (12 campioni) e `training.csv` (7.035 campioni).

**Prodotti.** `01_preprocessing.ipynb` (codice e verifiche numeriche), `preprocessing.py` (la
stessa trasformazione come funzione richiamabile, per il file d'esame), `manuale.csv` e
`training.csv` — entrambi con `USERID` come indice.

**Risultato.** 7.047 righe-studente, 8 feature, target `ABBANDONO` al 57,7% di positivi, zero
valori mancanti.

---

## 0. Il dataset di partenza

*act-mooc* (Stanford SNAP) è il registro delle azioni compiute da 7.047 studenti su una
piattaforma MOOC in circa 30 giorni: 411.749 azioni su 97 attività del corso. È distribuito in tre
file TSV con lo stesso numero di righe, riferiti alla stessa lista di azioni:

| file | colonne | contenuto |
|---|---|---|
| `mooc_actions.tsv` | `ACTIONID`, `USERID`, `TARGETID`, `TIMESTAMP` | chi ha fatto cosa e quando |
| `mooc_action_features.tsv` | `ACTIONID`, `FEATURE0..3` | quattro attributi numerici dell'azione |
| `mooc_action_labels.tsv` | `ACTIONID`, `LABEL` | `1` sull'azione dopo la quale lo studente abbandona |

Il dataset non arriva quindi in forma tabellare: è una sequenza di eventi. Il lavoro del Task 1 non
è pulire i dati — come si vedrà al punto 3 sono già puliti — ma **cambiare l'unità di analisi**,
passando dall'azione allo studente.

---

## 1. Unione dei tre file

**Cosa abbiamo fatto.** Un'unica tabella a livello azione (411.749 × 9), affiancando le colonne
`FEATURE0..3` e `LABEL` alle azioni **per posizione**, con `.to_numpy()`, e non con un `merge`.

**Perché.** `ACTIONID` è una chiave vera solo in due file su tre:

| file | righe | `ACTIONID` distinti | è chiave |
|---|---|---|---|
| `mooc_actions.tsv` | 411.749 | 411.749 | sì |
| `mooc_action_features.tsv` | 411.749 | 411.749 | sì |
| `mooc_action_labels.tsv` | 411.749 | 396.633 | **no** |

Nel file delle etichette 15.116 righe hanno `ACTIONID` ripetuto o discordante dalla propria
posizione. Un `merge(on="ACTIONID")` perderebbe quelle righe e ne duplicherebbe altrettante, in
silenzio. I tre file hanno però lo stesso numero di righe nello stesso ordine — è così che il
dataset è distribuito — quindi l'accostamento per posizione è l'unica lettura coerente.

**Alternative considerate.**
- `merge` anche per le etichette → scartato: perde e duplica righe senza segnalarlo.
- `drop_duplicates` sugli `ACTIONID` delle etichette → scartato: sceglierebbe arbitrariamente
  quale copia tenere.

**Cosa abbiamo verificato.** Righe e valori distinti dei tre file; `ACTIONID` identico fra azioni e
feature (`.equals()` restituisce `True`); 15.116 righe discordanti fra azioni ed etichette.

---

## 2. Controllo di qualità a livello azione

**Cosa abbiamo fatto.** Verificati valori mancanti, righe duplicate, range e tipi.

**Cosa abbiamo verificato.**

| controllo | esito |
|---|---|
| valori mancanti | 0 su tutte e 9 le colonne |
| righe duplicate | 0 |
| duplicati su `USERID` + `TARGETID` + `TIMESTAMP` | 54 coppie (108 righe, 37 studenti), di cui 31 identiche anche nelle `FEATURE` |
| `TIMESTAMP` negativi | 0 |
| valori assunti da `LABEL` | solo 0 e 1 |
| arco temporale | 29,77 giorni, coerente con la descrizione del dataset |
| `FEATURE0..3` | media 0 e deviazione standard 1 |

**Le 54 coppie, e perché le teniamo.** Il confronto va fatto sulle `FEATURE`, e l'esito è meno
netto di quanto sembri: **23 coppie su 54** hanno `FEATURE0..3` diverse, e sono davvero due gesti
distinti caduti nello stesso secondo, dato che i `TIMESTAMP` hanno risoluzione di un secondo. Le
altre **31 sono righe indistinguibili**: stesse feature e stessa `LABEL`, cambia solo `ACTIONID`.

Le abbiamo tenute comunque, anche le 31. Il dataset numera le azioni una per una e non dichiara
alcuna de-duplicazione: nulla permette di stabilire se una coppia identica sia un doppione tecnico
o un gesto realmente ripetuto, e le feature dell'azione non contengono l'informazione che
servirebbe a distinguerli. Scartarne una a caso sarebbe una scelta arbitraria travestita da
pulizia. L'effetto sui risultati è comunque nullo: 62 righe su 411.749, in 23 studenti, di cui
4 abbandoni.

**Sulle `FEATURE0..3`.** Sono già standardizzate all'origine (media 0, deviazione 1), quindi in
questa fase non serve alcuna trasformazione di scala. Lo scaling che servirà ai modelli del Task 5
verrà comunque stimato **dentro una `Pipeline`, sulle sole pieghe di addestramento**, e mai sui
file salvati: applicarlo qui significherebbe far vedere al preprocessing anche i dati di test.

---

## 3. L'unità di analisi è lo studente, non l'azione

**Cosa abbiamo fatto.** Da 411.749 righe-azione a 7.047 righe-studente, con target
`ABBANDONO = groupby("USERID")["LABEL"].max()`.

**Perché.** `LABEL = 1` marca l'**ultima azione** di chi abbandona, non una proprietà dell'azione in
sé. A livello azione la domanda predittiva diventerebbe «questa riga è l'ultima del suo gruppo?»,
una domanda già risolta dall'ordinamento dei dati e priva di interesse. La consegna del dataset
chiede di prevedere l'abbandono *dello studente*: l'unica formulazione onesta è «questo studente
abbandonerà?», e la sua unità di analisi è lo studente.

**Cosa abbiamo verificato.**
- 4.066 azioni su 4.066 con `LABEL = 1` sono l'ultima azione del proprio studente.
- Nessuno studente ha più di un'azione etichettata (i valori di `LABEL` sommati per studente sono
  solo 0 e 1).
- 7.047 studenti distinti, 4.066 dei quali positivi.

**Conseguenza sullo sbilanciamento.** La prevalenza passa dallo **0,99% a livello azione al 57,7%
a livello studente**: cambia il denominatore (411.749 → 7.047), il numeratore resta 4.066. Ogni
studente che abbandona porta con sé una sola azione positiva e decine di azioni negative.

Le classi sono quindi **sostanzialmente bilanciate** e nei task successivi non serve alcun
riequilibrio: né sovracampionamento, né SMOTE, né pesi di classe. Lo sbilanciamento estremo
descritto nella scheda del dataset è un effetto dell'unità di analisi sbagliata, non una proprietà
del problema.

---

## 4. Esclusione dell'ultima azione di ogni studente

**Cosa abbiamo fatto.** Ordinate le azioni per `USERID, TIMESTAMP, ACTIONID`, tolta l'ultima azione
di **ogni** studente e calcolate le feature sulle 404.702 azioni rimaste.

**Perché.** L'ultima azione è la riga su cui l'etichetta è definita: usarla anche come ingresso
significherebbe descrivere lo studente con il gesto stesso che ne ha determinato la classe.

**Perché per tutti e non solo per chi abbandona.** Se togliessimo l'ultima azione solo ai positivi,
la regola dipenderebbe dal target e introdurrebbe una differenza sistematica fra le due classi (i
positivi descritti da *n−1* azioni, i negativi da *n*). Applicata a tutti, la regola è cieca
rispetto alla classe.

**Cosa abbiamo verificato.** 411.749 − 7.047 = 404.702 azioni rimaste; tutti i 7.047 studenti
sopravvivono, e il più povero conserva 4 azioni (il minimo del dataset è 5 azioni per studente).

---

## 5. Le otto feature

**Cosa abbiamo fatto.** Riassunto la storia di ogni studente in otto feature, tutte numeriche:

| feature | definizione | significato |
|---|---|---|
| `n_azioni` | numero di righe del gruppo | quanto è stato attivo |
| `n_attivita_distinte` | `TARGETID` distinti | su quante delle 97 attività si è mosso |
| `n_giorni_attivi` | giorni distinti (`TIMESTAMP // 86400`) | in quanti giorni diversi si è collegato |
| `durata_giorni` | (`TIMESTAMP` max − min) / 86400 | quanto è durata la sua permanenza |
| `feature0_media` … `feature3_media` | media di `FEATURE0..3` | il profilo medio delle sue azioni |
| `ABBANDONO` | `max(LABEL)` | **target** |

Tutti i valori sono arrotondati a 3 decimali, per leggibilità e perché la terza cifra è ampiamente
sotto la risoluzione utile delle grandezze in gioco.

**Perché queste.** Le prime quattro rispondono alle domande naturali su uno studente — quanto,
quanto a lungo, quanto spesso, quanto in ampiezza — e sono immediatamente interpretabili in un
grafico e in un albero di decisione, cosa che conta per i Task 2, 3 e 4. Le ultime quattro sono
l'unica informazione descrittiva dell'azione presente nel dataset: la media è il modo più semplice
di portarla a livello studente senza moltiplicare le colonne.

**Alternative considerate.**
- `intervallo_medio_ore` (media dei tempi fra azioni consecutive) → scartata: è esattamente
  `durata_giorni / (n_azioni − 1)`, cioè una funzione deterministica di due feature già presenti.
- `intervallo_mediano_ore` → scartata: le azioni arrivano a raffiche, la mediana degli intervalli
  vale 22 secondi (quartili 12,5 e 51,5 secondi) e non separa le classi: da sola dà AUC 0,50.
- somma e massimo di `FEATURE0..3` accanto alla media → scartati: raddoppierebbero le colonne
  restando in buona parte ridondanti (la somma correla con la media fra 0,66 e 0,91 a seconda della
  feature), in un progetto che deve restare leggibile.

**`USERID` non è una feature.** È un identificativo, e resta come indice solo per poter risalire
allo studente. Nei task successivi i file vanno letti con `index_col="USERID"` e le feature sono
`X = df.drop(columns="ABBANDONO")`.

**Cosa abbiamo verificato.** 7.047 righe × 9 colonne, tutte numeriche; 0 valori mancanti, 0 valori
infiniti. Le 256 righe duplicate non sono un errore: sono studenti diversi (`USERID` è unico) con
lo stesso profilo, tipicamente chi ha compiuto pochissime azioni in un solo giorno, per cui le
feature collassano sugli stessi valori. Rappresentano studenti reali e vanno tenute.

---

## 6. `manuale.csv` — 12 campioni

**Cosa abbiamo fatto.** 6 studenti che hanno abbandonato e 6 che hanno proseguito, estratti con
`sample(random_state=42)` e poi mescolati.

**Perché 12 e bilanciati.** La consegna chiede 10–15 campioni. Il bilanciamento serve perché su un
campione così piccolo una proporzione 7/5 sposterebbe già in modo visibile le probabilità a priori
calcolate a mano al Task 2; 6+6 le rende esattamente 0,5 e 0,5, e i conti restano gestibili su
carta.

**Perché casuali e non scelti.** Scegliere «i casi più chiari» produrrebbe un file su cui qualunque
classificatore funziona, e quindi una valutazione priva di significato. Il seme fisso (42) rende
comunque l'estrazione riproducibile.

---

## 7. `training.csv` — 7.035 campioni

**Cosa abbiamo fatto.** Tutti gli studenti tranne i 12 di `manuale.csv`.

**Perché tutti.** La consegna lascia libera la dimensione. 7.035 righe con 8 feature sono già poche
per gli standard dei modelli visti a lezione: campionarne un sottoinsieme butterebbe via
informazione senza far risparmiare tempo di calcolo apprezzabile — l'intera pipeline gira in
pochi secondi.

**Perché i 12 di `manuale.csv` sono esclusi.** Saranno già stati usati per costruire a mano i
classificatori del Task 2: lasciarli anche qui li renderebbe dati già visti nella valutazione dei
Task 4 e 5.

**Nota.** La separazione fra training set e test set **non** avviene qui: avviene al Task 5, dentro
`train_test_split`, come prescrive la consegna. `training.csv` è il dataset preprocessato completo,
non il solo insieme di addestramento.

**Cosa abbiamo verificato.** 7.035 righe; prevalenza 57,71%, identica a quella del dataset
completo; intersezione vuota con `manuale.csv`; 7.035 + 12 = 7.047; il file riletto da disco è
identico alla tabella in memoria.

---

## 8. Verifica di sanità e nota critica

**Cosa abbiamo fatto.** Una regressione logistica con parametri di default, 5 pieghe, sulle 8
feature di `training.csv`. Non è la modellazione del Task 5: è un termometro per controllare che il
dataset costruito contenga segnale.

| | valore |
|---|---|
| AUC (5 pieghe) | 0,877 |
| accuratezza (5 pieghe) | 0,797 |
| accuratezza predicendo sempre la classe maggioritaria | 0,577 |

**Nota critica, da riprendere ai Task 4 e 5.** Il segnale viene quasi tutto dalle quattro feature di
attività, tutte correlate **negativamente** con il target (da −0,55 a −0,59): chi abbandona ha una
storia più corta e più povera. In parte questo è esattamente il fenomeno da prevedere; in parte è
un effetto di come l'etichetta è costruita, perché un abbandono è per definizione la fine
dell'attività di uno studente. Un modello che impara «poche azioni ⇒ abbandono» sta quindi
sfruttando una regolarità in parte tautologica, e questo va detto quando se ne commenteranno le
prestazioni.

**L'alternativa che abbiamo provato e scartato.** Calcolare le feature su una **finestra iniziale**
di azioni elimina alla radice il problema: guardano solo l'inizio della storia e non possono sapere
quanto durerà. La formulazione diventa «prevedere l'abbandono dai primi gesti», che è anche più
utile in pratica. I numeri che seguono sono rigenerati dall'appendice del notebook.

La finestra va presa su `storia`, cioè **dopo** aver tolto l'ultima azione. Pescarla dal dataset
completo rimetterebbe dentro proprio la riga su cui l'etichetta è definita, per ogni studente la
cui storia sta interamente nella finestra — lo stesso difetto corretto al punto 4. Il minimo di
`storia` è 4 azioni, quindi **4 è l'ampiezza massima che non esclude nessuno**. Il risultato è che
il segnale sparisce quasi del tutto: **AUC 0,535** contro 0,877.

Allargare la finestra non aiuta, e costa caro. Con 10 azioni vanno esclusi i 1.377 studenti che ne
hanno di meno, e sono **1.276 positivi contro 101 negativi** — chi abbandona presto ha per forza
una storia corta: la prevalenza scende dal 57,7% al 49,2%, cioè la selezione finisce governata dal
target, proprio il difetto che la finestra doveva evitare. E anche a quel prezzo l'AUC resta
**0,542**.

Abbiamo scelto la storia completa perché è l'unica che consegna ai task successivi un dataset su
cui si possa effettivamente lavorare, dichiarando apertamente il limite invece di nasconderlo.

---

## 9. `preprocessing.py` e il file d'esame

**Cosa abbiamo fatto.** Impacchettato la trasformazione in un modulo con due funzioni,
`carica_azioni` e `costruisci_studenti`, più una scorciatoia `carica_per_predire` che restituisce
`(X, y)` pronti per il classificatore. Il notebook resta la derivazione passo per passo; il modulo
è la stessa trasformazione resa richiamabile.

**Perché.** La consegna prevede che in sede d'esame venga fornito **`real_settings.csv`**, per
testare il classificatore scelto al Task 5. Le otto feature di `training.csv` sono definite da noi
e **non esistono nel dataset originale**: il docente non può quindi consegnare un file che le
contenga già. Quel file arriverà con ogni probabilità nel formato nativo di act-mooc — un log di
azioni — e andrà trasformato sul momento. Senza un modulo, quella trasformazione andrebbe
ricostruita a mano dalle celle del notebook, davanti al docente.

**Come si comporta.** `costruisci_studenti` riconosce da sola il formato ricevuto:

| il file ricevuto è… | cosa fa |
|---|---|
| un log di azioni con `LABEL` | costruisce le 8 feature **e** il target `ABBANDONO` |
| un log di azioni senza `LABEL` | costruisce le sole 8 feature, identiche al caso etichettato |
| già una tabella a livello studente | la restituisce invariata, con `USERID` come indice |

Le regole del Task 1 valgono anche lì: l'ultima azione di ogni studente viene esclusa pure sul file
d'esame, altrimenti le sue feature non sarebbero confrontabili con quelle su cui il modello è stato
addestrato. Uno studente con una sola azione resterebbe quindi senza feature, e la funzione lo
segnala esplicitamente invece di farlo sparire in silenzio (nel nostro dataset non accade: il
minimo è 5 azioni per studente).

**Alternative considerate.**
- **Tenere le colonne originali in `training.csv`**, una riga per azione, così che il file d'esame
  entri direttamente nel modello → scartato, ed è la scelta più importante di tutto il Task 1.
  A livello azione il problema non è apprendibile: con le sole colonne del dataset
  (`TARGETID`, `TIMESTAMP`, `FEATURE0..3`) una regressione logistica ottiene **F1 = 0,000** e una
  *average precision* di 0,031 contro lo 0,0099 di un classificatore casuale — cioè non predice
  mai un abbandono, pur mostrando un'accuratezza del 99%. Il motivo è che `LABEL` non è una
  proprietà della riga ma della sua **posizione** nella sequenza: la regola «è l'ultima azione
  dello studente», che non impara nulla, ricostruisce tutte e 4.066 le etichette (recall 1,000).
  Tenere il formato originale significherebbe non fare il Task 1, che la consegna definisce come
  «trasformare il dataset in un formato comprensibile dal classificatore».
- **Duplicare il codice del notebook in uno script separato** → scartato: due copie divergono. La
  sezione 10 del notebook verifica con `.equals()` che il modulo produca esattamente la stessa
  tabella, così una divergenza si vede subito.

**Limite dichiarato.** `n_azioni` e `durata_giorni` dipendono dall'ampiezza della finestra di
osservazione, che nel nostro dataset è di 29,77 giorni. Se `real_settings.csv` coprisse un arco
temporale diverso, quelle feature cambierebbero scala e le soglie apprese dal modello si
sposterebbero. È la stessa tautologia discussa al punto 8, vista dal lato della predizione.

**Cosa abbiamo verificato.** Il modulo riproduce la tabella del notebook e `training.csv` con
`.equals()` a `True`; le tre modalità d'ingresso danno le stesse otto colonne nello stesso ordine;
un log privo di `LABEL` produce esattamente le stesse feature di uno etichettato; il classificatore
addestrato su `training.csv` accetta l'uscita senza alcun adattamento.

---

## Riepilogo

| passo | risultato |
|---|---|
| unione dei tre file | 411.749 azioni × 9 colonne, unite per posizione |
| controllo qualità | 0 mancanti, 0 duplicati, 0 valori fuori range |
| cambio di unità di analisi | da 411.749 azioni a 7.047 studenti |
| target | `ABBANDONO`, 57,7% di positivi (contro lo 0,99% a livello azione) |
| feature | 8 per studente, calcolate escludendo l'ultima azione di ognuno |
| `manuale.csv` | 12 studenti, 6 per classe |
| `training.csv` | 7.035 studenti |
| `preprocessing.py` | la stessa trasformazione richiamabile sul file d'esame |

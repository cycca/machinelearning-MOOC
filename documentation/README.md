# Documentazione

Le motivazioni delle scelte. I notebook in `../notebooks/` contengono il codice e i numeri che le
verificano.

| documento | task | contenuto |
|---|---|---|
| [relazione.md](relazione.md) | tutti | **la relazione del progetto**: filo conduttore, i cinque task, limiti, riepilogo dei numeri |
| [01_preprocessing.md](01_preprocessing.md) | 1 | dal livello azione al livello studente, le 8 feature, i due CSV |
| [02.1_naive_bayes.md](02.1_naive_bayes.md) | 2 | primo classificatore manuale |
| [02.2_decision_tree.md](02.2_decision_tree.md) | 2 | secondo classificatore manuale |
| [03_analisi_esplorativa.md](03_analisi_esplorativa.md) | 3 | controlli di qualità, distribuzioni, correlazioni |
| [04_valutazione.md](04_valutazione.md) | 4 | ottimizzazione dei classificatori manuali |
| [05_modellazione.md](05_modellazione.md) | 5 | modelli Scikit-Learn e classificatore finale |
| [2know.md](2know.md) | — | appunti di preparazione all'orale: non fa parte della relazione |

Il Task 2 è diviso in due perché il gruppo è di due componenti e la consegna chiede *«uno o due
classificatori a seconda del numero di componenti»*.

`relazione.md` è il documento che espone il progetto: una sezione per task, con le decisioni
prese, la loro motivazione e la verifica numerica che le sostiene, più i limiti, la procedura per
il file d'esame e il riepilogo dei numeri. È il file da proiettare. Gli altri documenti restano la
fonte di dettaglio: la relazione li riordina, non li sostituisce.

`2know.md` **non fa parte della relazione**: raccoglie gli appunti di preparazione — come esporre,
cosa dire per primo, e le venticinque domande probabili con la risposta pronta. Sta qui per non
essere disperso, ma è materiale privato da leggere prima, non durante.

## Schema delle sezioni

```
## N. Titolo
**Cosa.**        le operazioni
**Perché.**      la motivazione
**Verificato.**  i controlli stampati dal notebook
```

Dove una scelta ne escludeva altre, c'è **Alternative scartate** con una riga per alternativa.

## Regole valide per tutti i task

1. **`USERID` non è mai una feature.** È un identificativo e resta come indice.
2. **Lo scaling si stima dentro una `Pipeline`**, sulle sole pieghe di addestramento. Mai sui file
   salvati, mai sul file d'esame.
3. **Le classi sono bilanciate** (57,7%): nessun SMOTE, nessun peso di classe. Lo sbilanciamento
   estremo esiste solo a livello azione, cioè nell'unità di analisi sbagliata.
4. **Il file d'esame passa da `preprocessing.costruisci_studenti`**, perché le 8 feature non
   esistono nel dataset originale.
5. **Nessun ciclo Python sui dati.** Ogni calcolo che tocca righe, colonne o soglie candidate è
   scritto come operazione su array o su `DataFrame`. I `for` rimasti scorrono modelli, pieghe di
   cross-validation, iperparametri o assi di un grafico — cose che non si possono vettorizzare
   perché ogni giro fa una cosa diversa. La motivazione, con la tabella dei `for` legittimi, è
   in «Come è scritto il codice» di [relazione.md](relazione.md).

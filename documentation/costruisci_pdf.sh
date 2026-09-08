#!/usr/bin/env bash
# Assembla tutta la documentazione in un unico PDF.
#
# Uso:   ./costruisci_pdf.sh [percorso_di_uscita]
# Default: ../../progetto_ML_2026.pdf, cioe' in machine_learning/, fuori dal progetto:
# il PDF e' un artefatto generato, i sorgenti sono i .md qui accanto.
#
# Richiede:  sudo dnf install pandoc-cli python3-weasyprint
#
# Stile, frontespizio e template HTML sono incorporati qui sotto: sono usati solo da
# questo script, quindi tenerli in file separati aggiungeva cartelle senza vantaggi.
set -euo pipefail
cd "$(dirname "$0")"

DOCUMENTI=(00_sintesi.md 01_preprocessing.md 02.1_naive_bayes.md 02.2_decision_tree.md
           03_analisi_esplorativa.md 04_valutazione.md 05_modellazione.md)
USCITA=${1:-../../progetto_ML_2026.pdf}

TEMP=$(mktemp -d)
trap 'rm -rf "$TEMP"' EXIT

cat > "$TEMP/stile.css" <<'FINE_CSS'
@page {
  size: A4;
  margin: 2.2cm 1.9cm 2.0cm 1.9cm;
  @bottom-center { content: counter(page); font: 9pt "DejaVu Serif", serif; color: #666; }
}
@page :first { margin: 0; @bottom-center { content: none; } }

html, body { margin: 0; padding: 0; max-width: none; width: auto; }
body { font: 10.5pt/1.45 "DejaVu Serif", Georgia, serif; color: #1a1a1a; hyphens: auto; }

/* --- frontespizio --- */
#frontespizio { page-break-after: always; padding: 6cm 2.6cm 0 2.6cm; }
#frontespizio h1 { font-size: 27pt; line-height: 1.22; margin: 0 0 .5cm 0; border: none;
  padding: 0; page-break-before: avoid; hyphens: none; }
#frontespizio .sottotitolo { font-size: 12.5pt; color: #555; line-height: 1.5; margin: 0 0 3cm 0; }
#frontespizio table.meta { font-size: 10.5pt; width: 100%; border-collapse: collapse; }
#frontespizio table.meta tr { border: none; }
#frontespizio table.meta td { padding: .16cm .2cm .16cm 0; border: none; vertical-align: top; }
#frontespizio table.meta td:first-child { color: #2f4f6f; font-weight: bold; width: 4.4cm;
  white-space: nowrap; }
.titolo-indice { page-break-before: avoid; }

/* --- titoli --- */
h1 { font-size: 17pt; color: #2f4f6f; border-bottom: 2px solid #2f4f6f; padding-bottom: .18cm;
     margin: 0 0 .55cm 0; page-break-before: always; page-break-after: avoid; }
h2 { font-size: 13pt; color: #2f4f6f; margin: .75cm 0 .25cm 0; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: .55cm 0 .2cm 0; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
hr { display: none; }

/* --- indice --- */
#TOC { page-break-after: always; }
#TOC ul { list-style: none; padding-left: 0; }
#TOC ul ul { padding-left: .8cm; font-size: 9.5pt; color: #444; }
#TOC li { margin: .12cm 0; }
#TOC a { text-decoration: none; color: #1a1a1a; }
#TOC a::after { content: leader(".") target-counter(attr(href), page); color: #666; }
#TOC > ul > li > a { font-weight: bold; color: #2f4f6f; }

/* --- tabelle --- */
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: .35cm 0;
        page-break-inside: avoid; }
th { background: #eef2f6; text-align: left; border-bottom: 1.5px solid #2f4f6f; }
th, td { padding: .13cm .22cm; vertical-align: top; }
tbody tr { border-bottom: .5px solid #d5dde5; }

/* --- codice --- */
code { font: 9pt "DejaVu Sans Mono", monospace; background: #f2f4f7; padding: .03cm .12cm;
       border-radius: 2px; }
pre { background: #f7f8fa; border-left: 3px solid #9fb3c8; padding: .28cm .35cm;
      font-size: 8.5pt; line-height: 1.35; overflow-wrap: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }

/* --- figure --- */
img { max-width: 100%; display: block; margin: .35cm auto .1cm auto; page-break-inside: avoid; }
figure { page-break-inside: avoid; margin: .4cm 0; }
figcaption { font-size: 8.5pt; color: #555; text-align: center; font-style: italic; }

blockquote { border-left: 3px solid #9fb3c8; margin: .35cm 0; padding: .1cm .45cm;
             color: #333; font-style: italic; }
FINE_CSS

cat > "$TEMP/modello.html" <<'FINE_TEMPLATE'
<!DOCTYPE html>
<html lang="$lang$">
<head>
  <meta charset="utf-8">
  <title>$title$</title>
$for(header-includes)$
$header-includes$
$endfor$
</head>
<body>
$for(include-before)$
$include-before$
$endfor$
$if(toc)$
<nav id="TOC"><h1 class="titolo-indice">Indice</h1>
$table-of-contents$
</nav>
$endif$
$body$
</body>
</html>
FINE_TEMPLATE

cat > "$TEMP/frontespizio.html" <<'FINE_FRONTESPIZIO'
<div id="frontespizio">
  <h1>Previsione dell&rsquo;abbandono in una piattaforma MOOC</h1>
  <p class="sottotitolo">Progetto di Fondamenti e Applicazioni del Machine Learning<br>Anno Accademico 2026</p>
  <table class="meta">
    <tr><td>Studenti</td><td>Federico Ciccarelli &middot; Lorenzo Lallone</td></tr>
    <tr><td>Docenti</td><td>prof. Fabrizio Rossi &middot; prof. Fabio Persia</td></tr>
    <tr><td>Dataset</td><td>act-mooc (Stanford SNAP) &mdash; 411.749 azioni, 7.047 studenti</td></tr>
    <tr><td>Task</td><td>classificazione binaria &mdash; abbandono dello studente</td></tr>
    <tr><td>Modello finale</td><td>regressione logistica &mdash; 79,25% di accuratezza sul test set</td></tr>
    <tr><td>Repository</td><td>github.com/cycca/machinelearning-MOOC</td></tr>
  </table>
</div>
FINE_FRONTESPIZIO

pandoc "${DOCUMENTI[@]}" \
  --standalone --toc --toc-depth=2 --from=gfm --to=html5 \
  --metadata title="Progetto di Machine Learning 2026" \
  --metadata lang=it \
  --template="$TEMP/modello.html" \
  --include-in-header=<(printf '<style>\n%s\n</style>\n' "$(cat "$TEMP/stile.css")") \
  --include-before-body="$TEMP/frontespizio.html" \
  -o "$TEMP/assemblato.html"

# il file HTML va generato qui dentro, cosi' i percorsi relativi a figure/ si risolvono
cp "$TEMP/assemblato.html" .assemblato.html
python3 -m weasyprint .assemblato.html "$USCITA" 2>/dev/null
rm -f .assemblato.html
echo "prodotto: $(realpath "$USCITA")  ($(du -h "$USCITA" | cut -f1))"

# ZUSAMMENFASSUNG 02: Word Embeddings & Recurrent Neural Networks (ERWEITERT)

## Übersicht
- **Seitenzahl:** ~36 Seiten (AdvancedML-02-WordEmbeddings-RNNs.pdf)
- **Hauptthemen:** Word Embeddings (Word2Vec, FastText, BPE), RNN, LSTM, GRU, Seq2Seq, Attention
- **Prüfungsrelevanz:** 🔴 SEHR HOCH - Grundlagen für alle NLP-Modelle

---

## 1. Darstellungsmöglichkeiten von Text

### 1.1 Bag of Words (BoW) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Grundidee
Jedes Wort wird als **One-Hot-Vektor** repräsentiert. Die Dimension entspricht der Größe des Vokabulars.

**Formale Definition:**
- Vokabular V = {w₁, w₂, ..., w_|V|}
- One-Hot-Vektor für Wort wᵢ: vᵢ ∈ ℝ^|V| mit vᵢ[j] = 1 wenn j=i, sonst 0

**Beispiel:**
```
Input: "Language technology is awesome"
Vokabular: {awesome:0, technology:1, is:2, hello:3, world:4, language:5}

One-Hot-Vektoren:
"awesome"    → [1, 0, 0, 0, 0, 0]
"technology" → [0, 1, 0, 0, 0, 0]
"is"         → [0, 0, 1, 0, 0, 0]
"language"   → [0, 0, 0, 0, 0, 1]

BoW-Vektor für Satz: [1, 1, 1, 0, 0, 1]  (zählt Vorkommen)
```

#### Eigenschaften von BoW
- **Dimension:** |V| (oft 30.000-100.000+ bei großen Korpora)
- **Sparsity:** Die meisten Einträge sind 0 (sparse)
  - Typischerweise <1% der Einträge ≠ 0
- **Verlust der Wortreihenfolge:** "Dog bites man" = "Man bites dog"

#### Probleme von BoW
1. **Keine Semantik:** Alle Wörter sind gleich weit voneinander entfernt
   - cos("King", "Queen") = cos("King", "Table") = 0
2. **Keine Wortreihenfolge:** BoW ignoriert Sequenzinformation
3. **Hoher Speicherbedarf:** Bei |V|=100.000 benötigt jeder Vektor 400 KB
4. **Data Sparsity:** Seltene Wörter haben wenige Beispiele

---

### 1.2 TF-IDF (Term Frequency - Inverse Document Frequency) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Motivation
Häufige Wörter ("und", "der", "the", "is") sollten nicht dominieren. TF-IDF gewichtet Wörter nach ihrer **Diskriminierungskraft**.

#### Formeln ⭐ AUSWENDIG LERNEN

**Term Frequency (TF):**
```
TF(w, d) = (#w in d) / (#Wörter in d)
```
- Lokale Wichtigkeit im Dokument d
- Alternative: log(1 + count(w,d)) zur Dämpfung

**Inverse Document Frequency (IDF):**
```
IDF(w, D) = log(|D| / #{d ∈ D: w ∈ d})
```
- |D| = Gesamtzahl der Dokumente im Korpus
- #{d ∈ D: w ∈ d} = Anzahl Dokumente, die w enthalten
- Globale Seltenheit im Korpus

**TF-IDF:**
```
TF-IDF(w, d, D) = TF(w, d) × IDF(w, D)
```

#### Beispielrechnung
```
Korpus: 3 Dokumente
D1: "The cat sat on the mat"
D2: "The dog sat on the log"
D3: "The cat and the dog are friends"

Für Wort "cat":
- TF("cat", D1) = 1/6 ≈ 0.167
- IDF("cat") = log(3/2) ≈ 0.405
- TF-IDF("cat", D1) = 0.167 × 0.405 ≈ 0.068

Für Wort "the":
- TF("the", D1) = 2/6 ≈ 0.333
- IDF("the") = log(3/3) = log(1) = 0
- TF-IDF("the", D1) = 0.333 × 0 = 0  (Stopwort wird eliminiert!)
```

#### Anwendung
- **Dokumentenvergleich:** Cosine Similarity zwischen TF-IDF-Vektoren
- **Information Retrieval:** Ranking von Dokumenten nach Relevanz
- **Input für ML-Modelle:** Features für Klassifikation, Clustering

**Cosine Similarity:**
```
cos(v₁, v₂) = (v₁ · v₂) / (|v₁| |v₂|)
```
- Wertebereich: [-1, 1], bei TF-IDF typisch [0, 1]
- 1 = identische Richtung (sehr ähnlich)
- 0 = orthogonal (keine Ähnlichkeit)

---

### 1.3 Vergleich: BoW/TF-IDF vs. Word2Vec ⭐ PRÜFUNGSRELEVANT

| Eigenschaft | BoW/TF-IDF | Word2Vec |
|-------------|------------|----------|
| **Dimension** | |V| (100.000+) | d (100-300) |
| **Sparsity** | Sparse (meist 0) | Dense (alle Werte ≠ 0) |
| **Semantik** | Keine | Ja (durch Training) |
| **Speicherbedarf** | Hoch (400 KB) | Niedrig (1.2 KB) |
| **Wortähnlichkeit** | Nicht erfassbar | Vektorarithmetik möglich |
| **Geeignet für** | Lineare Modelle (SVM, NB) | Neuronale Netze |

**Speicherbedarf im Vergleich:**
```
BoW bei |V|=100.000: 100.000 × 4 Bytes = 400 KB pro Vektor
Word2Vec bei d=300: 300 × 4 Bytes = 1.2 KB pro Vektor

Ersparnis: Faktor ~333 bei vergleichbarer oder besserer Performance!
```

---

## 2. Word Embeddings & Distributed Semantics

### 2.1 Distributional Hypothesis ⭐⭐⭐ PRÜFUNGSRELEVANT

**Zitat:** "You shall know a word by the company it keeps" (J.R. Firth, 1957)

**Kernidee:**
- Wörter mit ähnlicher Bedeutung treten in **ähnlichen Kontexten** auf
- Beispiel: "Katze" und "Hund" erscheinen beide in Kontexten wie:
  - "Das ___ schläft auf dem Sofa"
  - "Ich füttere den ___"
  - "Der ___ bellt/miaut"

**Ziel:**
- Ähnliche Wörter → Ähnliche Vektoren im ℝ^d
- d ≈ 100-300 (viel kleiner als Vokabulargröße)

**Beispiel Ongchoi (unbekanntes Wort):**
```
Beobachtungen:
- "Ongchoi schmeckt am besten mit Reis"
- "Anschließend Ongchoi mit Knoblauch sautieren"

Kontextwörter: Knoblauch, Reis, sautieren, schmeckt → Bereich "Kochen"
→ Ongchoi ist wahrscheinlich ein Gemüse/Kräutertyp
```

---

### 2.2 Word2Vec (Mikolov et al., 2013) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Zwei Architektur-Varianten

**1. CBOW (Continuous Bag-of-Words):**
```
Kontext:  "The ___ sat on the mat"
          [cat]

Input:    Mittelung über Kontextwort-Vektoren
          v("The") + v("sat") + v("on") + v("the") + v("mat")

Output:   Vorhersage: "cat"
```
- **Richtung:** Kontext → Zielwort
- **Input:** Kontextwörter (one-hot-encoded), gemittelt
- **Hidden Layer:** Single Layer ohne Aktivierung
- **Output:** Wahrscheinlichkeit über Vokabular (Softmax)
- **Vorteile:** Schnelleres Training, gut für häufige Wörter
- **Nachteile:** Verliert Wortordnung, schlechter für seltene Wörter

**2. Skip-gram:**
```
Input:    "cat" (one-hot)

Output:   Vorhersage der Kontextwörter:
          P("The"|cat), P("sat"|cat), P("on"|cat), ...
```
- **Richtung:** Zielwort → Kontext
- **Input:** Zentrales Wort (one-hot)
- **Output:** Wahrscheinlichkeiten für Kontextwörter
- **Vorteile:** Bessere Qualität für seltene Wörter, feinere semantische Nuancen
- **Nachteile:** Langsamer (mehrfache Vorhersagen), höhere Varianz

#### Training

**Daten:**
- Große Textkorpora (Wikipedia, Common Crawl, Google News)
- Typisch: 100 Mrd. - 1 Billion Wörter

**Kontextfenster:**
- Größe c (typisch: c=5-10)
- Für Wort wₜ: Kontext = {wₜ₋c, ..., wₜ₋₁, wₜ₊₁, ..., wₜ₊c}

**Netzwerk-Architektur:**
```
Input Layer (|V|) → Projection Layer (d) → Output Layer (|V|)
     one-hot           Embedding              Softmax
```

**Parameter:**
- **Embedding-Matrix M ∈ ℝ^(|V|×d):**
  - Jede Spalte = Embedding eines Wortes
  - Nach Training: M enthält gelernte Wortvektoren
- **Output-Matrix W ∈ ℝ^(d×|V|):**
  - Wird oft verworfen nach Training

**Loss Function (Skip-gram):**
```
L = -Σ_{(w_t, w_c) ∈ D} log P(w_c | w_t)

P(w_c | w_t) = softmax(v_{w_c}^T · v_{w_t})
             = exp(v_{w_c}^T · v_{w_t}) / Σ_{w'} exp(v_{w'}^T · v_{w_t})
```

#### Optimierungstricks

**1. Hierarchical Softmax:**
- Reduziert Komplexität von O(|V|) auf O(log|V|)
- Binary Tree über Vokabular
- Jeder Pfad = Folge von binären Entscheidungen

**2. Negative Sampling:**
- Statt alle |V| Wörter zu betrachten: Nur k negative Beispiele
- Update: Positives Beispiel + k zufällige negative Wörter
- Typisch: k=5-20

```
L = -log(σ(v_{pos}^T · v_{center})) - Σ_{i=1}^{k} log(σ(-v_{neg_i}^T · v_{center}))
```

#### Vektorarithmetik für Analogien ⭐ PRÜFUNGSRELEVANT
```
king    - man   + woman   ≈ queen
[0.82]  [0.51] [0.73]     [0.79]

Paris   - France + Germany ≈ Berlin
Rom     - Italy  + Spain   ≈ Madrid
```

**Mathematisch:**
```
v(king) - v(man) + v(woman) = v_result
v_result ist am ähnlichsten zu v(queen) (höchste cosine similarity)
```

#### Bias in Embeddings ⚠️ PRÜFUNGSRELEVANT

**Problem:** Embeddings verstärken gesellschaftliche Vorurteile

**Beispiele (Caliskan et al., 2017 - Science):**
```
Man   : Doctor   ≈ Woman : Nurse
He    : Programmer ≈ She : Homemaker
Europäische Namen : positiver Sentiment
Afroamerikanische Namen : negativer Sentiment
```

**WEAT (Word Embedding Association Test):**
- Misst Bias analog zum IAT (Implicit Association Test)
- Testet systematische Assoziationen zwischen Zielgruppen und Attributen

**Folgen:**
- Bias wird in downstream-Anwendungen übernommen
- Bewerbungs-Screening, Kreditvergabe, etc. können diskriminieren

**Gegenmaßnahmen:**
- Hard Debiasing (Bolukbasi et al., 2016): Projektion auf subspace
- Soft Debiasing: Regularisierung während Training
- Bewusstsein für Limitationen

---

### 2.3 FastText (Bojanowski et al., 2016) ⭐⭐ PRÜFUNGSRELEVANT

#### Kernidee
Wörter werden als **Bag of Character n-grams** repräsentiert.

**Beispiel für "where" mit n=3:**
```
Special tokens: < für Wortbeginn, > für Wortende

n-grams: <wh, whe, her, ere, re>
Plus ganzes Wort: <where>

Vollständige n-grams (Länge 3-6):
<wh, whe, wher, here, ere, re, e>
```

#### Embedding-Berechnung
```
v(w) = Σ_{g ∈ ngrams(w)} v_g

wobei v_g = Embedding des n-grams g
```

#### Vorteile gegenüber Word2Vec

1. **Bessere Embeddings für seltene Wörter:**
   - Seltene Wörter teilen n-grams mit häufigen Wörtern
   - "unbelievable" teilt n-grams mit "believe", "unable", etc.

2. **Out-of-Vocabulary (OOV) Behandlung:** ⭐ PRÜFUNGSRELEVANT
   - Unbekannte Wörter können zur Laufzeit embeddet werden
   - Zerlege in n-grams → summiere Embeddings
   - Word2Vec: OOV = Fehler oder <UNK>-Token

3. **Morphologie wird erfasst:**
   - "running", "runs", "ran" teilen n-grams
   - Grammatikalische Beziehungen implizit gelernt

#### Beispiel: OOV-Wort "unfriend"
```
Training: "friend", "unhappy", "friendly" im Vokabular
          "unfriend" NICHT im Vokabular

FastText: "unfriend" → <un, unf, nfr, fri, rie, ien, end, nd>
          v("unfriend") = Σ v(n-grams)
          Teilt "un" mit "unhappy", "fri" mit "friend", etc.

Ergebnis: Sinnvolles Embedding trotz OOV!
```

---

### 2.4 Character Embeddings

#### Grundidee
Einzelne Zeichen als Vektoren. Wort ist Sequenz aus Character-Embeddings.

**Architektur-Optionen:**

1. **CNN über Characters:**
   ```
   Wort: "hello"
   Chars: [h, e, l, l, o] → [v_h, v_e, v_l, v_l, v_o]
   CNN mit verschiedenen Filter-Größen (3,4,5)
   Max-Pooling über Zeit
   → Wort-Vektor
   ```

2. **RNN über Characters:**
   ```
   Chars sequentiell durch LSTM/GRU
   Letzter Hidden State = Wort-Vektor
   ```

3. **Hybrid: CharCNN + Word2Vec:**
   - Summiere Character- und Word-Embeddings
   - Beste von beiden Welten

#### Vorteile
- **Vollständig OOV-fähig:** Jedes Wort darstellbar
- **Keine Tokenisierung nötig:** Rohtext verarbeitbar
- **Robust gegen Rechtschreibfehler:** "helo" ≈ "hello"
- **Morphologie explizit:** Prefixe, Suffixe, Stämme

#### Nachteile
- **Längere Sequenzen:** "internationalization" = 20 Characters
- **Höhere Rechenlast:** Mehr Schritte pro Wort
- **Semantik weniger direkt:** "Hund" und "Dog" teilen keine Characters

---

### 2.5 Byte Pair Encoding (BPE) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Motivation
Kompromiss zwischen Word-Level und Character-Level.

**Problem Word-Level:**
- Riesiges Vokabular (50.000+)
- Viele OOV-Wörter

**Problem Character-Level:**
- Sehr lange Sequenzen
- Semantik schwer zu lernen

**Lösung BPE:** Subword-Tokenisierung mit kontrollierter Vokabulargröße

#### Algorithmus ⭐ AUSWENDIG LERNEN

**Initialisierung:**
```
Vokabular = Alle einzelnen Characters
Wörter repräsentiert als Character-Sequenzen
```

**Iterativer Prozess:**
```
1. Zähle alle benachbarten Symbol-Paare im Korpus
2. Finde häufigstes Paar (z.B. "e" + "r" → "er")
3. Füge neues Symbol zum Vokabular hinzu
4. Ersetze alle Vorkommen des Paares
5. Wiederhole bis Vokabulargröße erreicht (z.B. 10.000)
```

#### Beispiel

**Start-Korpus:**
```
low: 5x
lower: 2x
newest: 6x
widest: 3x
```

**Initial:**
```
Vokabular: {l, o, w, e, r, n, s, t, w, i, d}
Wörter: l-o-w, l-o-w-e-r, n-e-w-e-s-t, w-i-d-e-s-t
```

**Iteration 1:**
```
Häufigstes Paar: "e" + "s" (9x in newest, widest)
Neues Symbol: "es"
Vokabular: {..., es}
```

**Iteration 2:**
```
Häufigstes Paar: "es" + "t" (9x)
Neues Symbol: "est"
Vokabular: {..., es, est}
```

**Nach vielen Iterationen:**
```
Vokabular: {l, o, w, e, r, n, s, t, i, d, lo, low, er, low er, est, wid, widest, ...}

Tokenisierung:
"lower" → ["low", "er"]
"newest" → ["new", "est"]
"widest" → ["wid", "est"]
```

#### Eigenschaften

**Vokabulargröße:**
- Kontrollierbar (typisch: 4.000-60.000)
- Trade-off: Größer = weniger OOV, aber mehr Parameter

**Produktive Morphologie:**
- "unbelievable" → ["un", "believ", "able"]
- Teile werden wiederverwendet

**OOV-Behandlung:**
- Unbekannte Wörter in bekannte Subwords zerlegbar
- "ChatGPT" → ["Chat", "G", "PT"] (wenn im Vokabular)

#### WordPiece (Google)
- Erweiterung von BPE
- Wählt Merge basierend auf Likelihood statt Frequenz
- Verwendet in BERT, GPT

#### SentencePiece
- BPE direkt auf Rohtext (ohne vorherige Tokenisierung)
- Behandelt Whitespace als eigenes Symbol
- Sprachunabhängig

---

## 3. Recurrent Neural Networks (RNNs)

### 3.1 Warum RNNs? ⭐⭐⭐ PRÜFUNGSRELEVANT

**Sequentielle Daten überall:**
- Text: Wörter in Sätzen
- Sprache: Audio-Samples
- Zeitreihen: Börsenkurse, Sensordaten
- Video: Frames in Sequenz

**Reihenfolge trägt Information:**
```
"The dog bites man" ≠ "The man bites dog"
"Ich habe gegessen" ≠ "Gegessen habe ich" (andere Betonung)
```

**Problem klassischer Netze:**
- Fully Connected: Keine Zeitabhängigkeiten
- CNN: Lokale Muster, aber keine langen Abhängigkeiten
- Annahme: Inputs unabhängig (i.i.d.)

---

### 3.2 Vanilla RNN Architektur ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Rekurrente Verbindung

**Update-Gleichung:** ⭐ AUSWENDIG LERNEN
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
y_t = g(W_y · h_t + b_y)
```

**Variablen:**
- **h_t ∈ ℝ^d:** Hidden State zum Zeitpunkt t ("Gedächtnis")
- **x_t ∈ ℝ^m:** Input zum Zeitpunkt t (z.B. Word Embedding)
- **y_t ∈ ℝ^k:** Output zum Zeitpunkt t
- **W_h ∈ ℝ^(d×d):** Hidden-to-Hidden Gewichtsmatrix (rekurrent)
- **W_x ∈ ℝ^(d×m):** Input-to-Hidden Gewichtsmatrix
- **W_y ∈ ℝ^(k×d):** Hidden-to-Output Gewichtsmatrix
- **b_h, b_y:** Bias-Vektoren

**Aktivierungsfunktionen:**
- **tanh:** Wertebereich (-1, 1), zentriert um 0
- **ReLU:** Wertebereich [0, ∞), vermeidet Vanishing teilweise
- **g:** Task-abhängig (Softmax für Klassifikation, linear für Regression)

#### Unrolling über Zeit

**Zeitliche Entfaltung:**
```
t=0: h_0 = tanh(W_h · h_{-1} + W_x · x_0 + b_h)
t=1: h_1 = tanh(W_h · h_0 + W_x · x_1 + b_h)
t=2: h_2 = tanh(W_h · h_1 + W_x · x_2 + b_h)
...
t=T: h_T = tanh(W_h · h_{T-1} + W_x · x_T + b_h)
```

**Visualisierung:**
```
x_0 → [RNN] → h_0 → y_0
         ↓
x_1 → [RNN] → h_1 → y_1
         ↓
x_2 → [RNN] → h_2 → y_2
         ↓
       ...
```

#### Eigenschaften ⭐ PRÜFUNGSRELEVANT

**1. Variable Input-Länge:**
- Verarbeitet Sequenzen beliebiger Länge T
- Keine feste Input-Dimension nötig

**2. Parameter Sharing:**
- W_h, W_x, W_y in jedem Zeitschritt gleich
- Parameterzahl unabhängig von T
- Generalisierung über Positionen hinweg

**3. Sequentielle Berechnung:** ⭐ PRÜFUNGSRELEVANT
- h_t benötigt h_{t-1}
- **Keine Parallelisierung über Zeit möglich!**
- Langsam bei langen Sequenzen

#### Parameterzahl

**Beispiel:**
```
Input-Dimension: m = 100 (Word Embedding)
Hidden-Dimension: d = 256
Output-Dimension: k = 10 (Klassen)

Parameter:
W_x: 256 × 100 = 25.600
W_h: 256 × 256 = 65.536  (rekurrent!)
W_y: 10 × 256 = 2.560
b_h: 256
b_y: 10

Gesamt: 93.962 Parameter
```

**Unabhängig von Sequenzlänge T!**

---

### 3.3 Backpropagation Through Time (BPTT) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Grundidee
Fehler wird über Zeit rückwärts propagiert.

**Loss Function:**
```
L = Σ_{t=1}^{T} l(y_t, ŷ_t)

wobei l = Cross-Entropy für Klassifikation
          MSE für Regression
```

#### Gradientenberechnung

**Kettenregel über Zeit:**
```
∂L/∂W_h = Σ_{t=1}^{T} ∂L/∂h_t · ∂h_t/∂W_h

∂h_t/∂W_h = ∂h_t/∂h_{t-1} · ∂h_{t-1}/∂W_h + ∂h_t/∂W_h (direkt)
```

**Rekursive Abhängigkeit:**
```
∂L/∂h_t = ∂L/∂y_t · ∂y_t/∂h_t + ∂L/∂h_{t+1} · ∂h_{t+1}/∂h_t

∂h_{t+1}/∂h_t = diag(1 - tanh²(...)) · W_h^T
              = diag(1 - h_{t+1}²) · W_h^T
```

#### Vanishing Gradient Problem ⚠️⭐⭐⭐ PRÜFUNGSRELEVANT

**Ursache:**
```
∂h_t/∂h_{t-k} = Π_{i=0}^{k-1} ∂h_{t-i}/∂h_{t-i-1}
              = Π_{i=0}^{k-1} diag(1 - tanh²(...)) · W_h^T
```

**Problem:**
- tanh-Ableitung: (1 - tanh²(x)) ∈ (0, 1]
- Typisch ≈ 0.25 für große |x|
- Bei vielen Multiplikationen: Produkt → 0
- Gradient "verschwindet" für frühe Zeitschritte

**Folge:**
- Keine Updates für frühe Zeitpunkte
- Lange Abhängigkeiten nicht lernbar
- "Memory" funktioniert nicht

#### Exploding Gradient Problem

**Ursache:**
- Bei großen Werten in W_h: Produkt kann explodieren
- Gradient → ∞
- Numerische Instabilität

**Lösung: Gradient Clipping** ⭐ PRÜFUNGSRELEVANT
```
g = Gradient
threshold = 1.0 (hyperparameter)

if ||g|| > threshold:
    g = g · (threshold / ||g||)
```
- Begrenzt Gradientennorm auf Maximum
- Verhindert Exploding Gradients
- Keine Auswirkung auf Vanishing Gradients

#### Truncated BPTT

**Idee:**
- Begrenze Backpropagation auf k Zeitschritte
- Statt über gesamte Sequenz: Nur letzte k Schritte

**Algorithmus:**
```
For t = 1 to T:
    Forward: Berechne h_t
    Backward: Berechne Gradienten für t, t-1, ..., t-k
    Update: W_h, W_x, W_y
```

**Vorteile:**
- Weniger Speicherbedarf
- Stabilere Gradienten
- Online-Learning möglich

**Nachteile:**
- Lange Abhängigkeiten > k nicht lernbar
- Heuristische Wahl von k

---

### 3.4 RNN-Probleme im Überblick ⭐⭐⭐ PRÜFUNGSRELEVANT

| Problem | Ursache | Auswirkung | Lösung |
|---------|---------|------------|--------|
| **Vanishing Gradient** | tanh-Ableitung < 1 | Keine langen Abhängigkeiten | LSTM, GRU |
| **Exploding Gradient** | Große W_h-Werte | Numerische Instabilität | Gradient Clipping |
| **Information Bottleneck** | h_t muss alles speichern | Informationsverlust | Attention |
| **Sequenzielle Berechnung** | h_t hängt von h_{t-1} ab | Keine Parallelisierung | Transformer |

---

### 3.5 Stacked RNN

**Analog zu MLP mit mehreren Layern:**
```
Layer 1: h_t¹ = tanh(W_x¹ · x_t + W_h¹ · h_{t-1}¹ + b_h¹)
Layer 2: h_t² = tanh(W_x² · h_t¹ + W_h² · h_{t-1}² + b_h²)
...
```

**Erhöht Modellkapazität:**
- Zeitreihen: 1-2 Layer
- Sentiment Analysis: 2-3 Layer
- Spracherkennung: bis 8 Layer
- Machine Translation: bis 8 Layer

---

## 4. LSTM: Long Short-Term Memory ⭐⭐⭐ PRÜFUNGSRELEVANT

### 4.1 Kernidee

**Hauptsächliche Innovation (Hochreiter & Schmidhuber, 1997):**
- Expliziter **Memory-Mechanismus** mit Gates
- Separater **Cell State C_t** für Langzeitgedächtnis
- **Gates** regulieren Informationsfluss
- Gradienten können über viele Schritte fließen

**Vergleich Vanilla RNN vs. LSTM:**
```
RNN: Information muss durch tanh und W_h bei jedem Schritt
     → Gradient wird multipliziert mit W_h und tanh'-Ableitung

LSTM: Cell State hat additive Updates (keine Multiplikation)
      → Gradient kann direkt fließen (Constant Error Carousel)
```

---

### 4.2 LSTM-Komponenten im Detail ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Cell State (C_t)
- **Funktion:** Langzeitgedächtnis
- **Eigenschaft:** Additive Updates → Gradient fließt direkt
- **Dimension:** d (wie Hidden State)

#### Hidden State (h_t)
- **Funktion:** Kurzzeitgedächtnis / Output
- **Eigenschaft:** Wird an nächsten Schritt weitergegeben
- **Dimension:** d

#### Die 3 Gates ⭐ AUSWENDIG LERNEN

**1. Forget Gate (f_t):**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
- **Input:** [h_{t-1}, x_t] (Konkatenation von vorherigem Hidden State und Input)
- **Output:** f_t ∈ (0, 1)^d (elementweise)
- **Funktion:** Was aus C_{t-1} vergessen wird
  - f_t ≈ 0: Vergessen
  - f_t ≈ 1: Behalten

**2. Input Gate (i_t):**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```
- **i_t:** Welche neuen Informationen hinzufügen
- **C̃_t:** Kandidaten für neue Information (∈ (-1, 1))

**3. Output Gate (o_t):**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
```
- **Funktion:** Was aus C_t als h_t ausgegeben wird

---

### 4.3 LSTM Update-Gleichungen ⭐⭐⭐ PRÜFUNGSRELEVANT

**Schritt-für-Schritt:** ⭐ AUSWENDIG LERNEN

```
1. Forget Gate:
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. Input Gate:
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

3. Cell State Update:
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

4. Output Gate:
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(C_t)
```

**⊙ = Hadamard-Produkt (elementweise Multiplikation)**

**Wertebereiche:**
- σ (Sigmoid): (0, 1) - geeignet für Gates
- tanh: (-1, 1) - für Cell State Kandidaten und Output

---

### 4.4 LSTM Parameter-Berechnung ⭐⭐⭐ PRÜFUNGSRELEVANT

**Gegeben:**
```
Eingabedimension: d_x = 64
Hidden-Dimension: d_h = 128
```

**Pro Gate:**
```
W_x: d_h × d_x = 128 × 64 = 8.192
W_h: d_h × d_h = 128 × 128 = 16.384
Bias: d_h = 128

Pro Gate: 8.192 + 16.384 + 128 = 24.704 Parameter
```

**Gesamt (4 Gates: f, i, C̃, o):**
```
Gesamt: 4 × 24.704 = 98.816 Parameter
```

**Allgemeine Formel:** ⭐ AUSWENDIG LERNEN
```
N_params = 4 × d_h × (d_x + d_h + 1)

Beispiel: d_x=64, d_h=128
        = 4 × 128 × (64 + 128 + 1)
        = 4 × 128 × 193
        = 98.816 ✓
```

**Vergleich zu Vanilla RNN:**
```
RNN: d_h × (d_x + d_h + 1) = 128 × 193 = 24.704 Parameter

LSTM: ~4× mehr Parameter als RNN
      Aber: Deutlich bessere Performance bei langen Abhängigkeiten
```

---

### 4.5 Warum LSTM gegen Vanishing Gradient hilft ⭐⭐⭐ PRÜFUNGSRELEVANT

**Cell State Gradient:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

∂C_t/∂C_{t-1} = f_t + (Terme mit Ableitungen von f_t, i_t, C̃_t)
```

**Key Insight:**
- Der Term **f_t** geht direkt in den Gradienten ein
- Wenn f_t ≈ 1: Gradient fließt nahezu unverändert
- **Keine Multiplikation mit W_h bei jedem Schritt!**

**Constant Error Carousel:** ⭐ PRÜFUNGSRELEVANT
```
Wenn f_t = 1 und i_t = 0:
C_t = 1 ⊙ C_{t-1} + 0 ⊙ C̃_t = C_{t-1}

→ Information bleibt beliebig lange erhalten!
→ Gradient fließt ohne Dämpfung
```

---

### 4.6 LSTM-Varianten

#### Peephole Connections (Gers & Schmidhuber, 2000)
- Gates erhalten auch Zugriff auf C_{t-1}
```
f_t = σ(W_f · [h_{t-1}, x_t, C_{t-1}] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t, C_{t-1}] + b_i)
o_t = σ(W_o · [h_{t-1}, x_t, C_t] + b_o)
```

#### Coupled Gates
- Forget- und Input-Gate gekoppelt
```
f_t = 1 - i_t

→ Wenn ich neue Info hinzufüge (i_t=1), vergesse ich alte (f_t=0)
→ Weniger Parameter, ähnlich gute Performance
```

---

## 5. GRU: Gated Recurrent Unit ⭐⭐ PRÜFUNGSRELEVANT

### 5.1 Kernidee

**Motivation:**
- LSTM sehr erfolgreich, aber viele Parameter
- Frage: Brauchen wir Cell State + Hidden State?
- Antwort: Oft nicht! GRU reicht.

**Vorteile:**
- Weniger Parameter → schnelleres Training
- Einfacher zu implementieren
- Oft vergleichbare Performance zu LSTM

---

### 5.2 GRU-Komponenten ⭐ AUSWENDIG LERNEN

#### Update Gate (z_t)
```
z_t = σ(W_z · [h_{t-1}, x_t])
```
- **Kombiniert** Forget- und Input-Gate
- Bestimmt: Wie viel vom alten h_{t-1} behalten vs. wie viel neues h̃_t hinzufügen

#### Reset Gate (r_t)
```
r_t = σ(W_r · [h_{t-1}, x_t])
```
- Bestimmt: Wie viel vom alten Hidden State für Kandidaten verwenden
- r_t ≈ 0: Vergesse Vergangenheit für Kandidaten
- r_t ≈ 1: Volle Vergangenheit nutzen

#### Candidate Hidden State (h̃_t)
```
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
```
- **Wichtig:** r_t ⊙ h_{t-1} (elementweise Multiplikation)
- Reset Gate filtert alten Hidden State

#### Final Hidden State (h_t)
```
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```
- **Interpolation** zwischen alt und neu
- z_t ≈ 0: Behalte alten Zustand
- z_t ≈ 1: Übernehme neuen Kandidaten

---

### 5.3 GRU vs. LSTM ⭐ PRÜFUNGSRELEVANT

| Eigenschaft | GRU | LSTM |
|-------------|-----|------|
| **Gates** | 2 (Update, Reset) | 3 (Forget, Input, Output) |
| **States** | 1 (Hidden) | 2 (Cell + Hidden) |
| **Parameter** | Weniger | Mehr |
| **Training** | Schneller | Langsamer |
| **Performance** | Oft vergleichbar | Etwas besser bei sehr langen Abhängigkeiten |
| **Popularität** | Häufig in NLP | Sehr häufig, "Gold Standard" |

**GRU Parameter (d_x=64, d_h=128):**
```
Pro Gate (z, r): d_h × (d_x + d_h + 1) = 128 × 193 = 24.704
Candidate: d_h × (d_x + d_h + 1) = 24.704

Gesamt: 3 × 24.704 = 74.112 Parameter (mit Bias)

Vergleich LSTM: 98.816 Parameter
Ersparnis: ~25% weniger Parameter!
```

---

## 6. Bidirektionale RNNs/LSTMs/GRUs ⭐⭐ PRÜFUNGSRELEVANT

### 6.1 Grundidee

**Problem:**
- Unidirektional: Nur Kontext von links (Vergangenheit)
- Beispiel: "Ich wohne in ___" → Was kommt?

**Lösung:**
- Verarbeite Sequenz in **beide Richtungen**
- Links→Rechts: h_t^→ (vorwärts)
- Rechts→Links: h_t^← (rückwärts)

### 6.2 Architektur

```
Vorwärts-RNN:
h_t^→ = RNN(x_t, h_{t-1}^→)

Rückwärts-RNN:
h_t^← = RNN(x_t, h_{t+1}^←)

Kombination:
h_t = [h_t^→; h_t^←]  (Konkatenation, doppelte Dimension)
```

**Visualisierung:**
```
x_1 → [→RNN] → h_1^→
x_2 → [→RNN] → h_2^→
x_3 → [→RNN] → h_3^→

x_3 → [←RNN] → h_3^←
x_2 → [←RNN] → h_2^←
x_1 → [←RNN] → h_1^←

Output: h_1 = [h_1^→; h_1^←], h_2 = [h_2^→; h_2^←], ...
```

### 6.3 Eigenschaften

**Vorteile:**
- **Voller Kontext:** Vergangenheit + Zukunft
- **Bessere Performance:** Bei vielen Tasks (NER, POS-Tagging)

**Nachteile:**
- **Nicht online-fähig:** Benötigt gesamte Sequenz im Voraus
- **2× Parameter:** Zwei separate RNNs
- **2× Rechenzeit:** Vorwärts + Rückwärts

**Anwendung:**
- ✅ NER (Named Entity Recognition)
- ✅ POS-Tagging
- ✅ Sentiment Analysis (wenn gesamter Text verfügbar)
- ❌ Echtzeit-Übersetzung (wenn Sequenz noch kommt)
- ❌ Autoregressive Generation (Decoder)

---

## 7. Seq2Seq (Encoder-Decoder) ⭐⭐⭐ PRÜFUNGSRELEVANT

### 7.1 Anwendungsfälle

**Seq2Seq-Aufgaben:**
- **Maschinelle Übersetzung:** DE → EN
- **Text-Zusammenfassung:** Lang → Kurz
- **Dialog-Systeme:** Input → Response
- **Bildbeschreibung:** Bild → Text (mit CNN als Encoder)

**Nicht Seq2Seq:**
- Sentiment Classification (Seq → Single Label)
- NER (Seq → Seq, aber gleiche Länge)
- Clustering (Unsupervised)

---

### 7.2 Architektur ⭐ AUSWENDIG LERNEN

#### Encoder
```
Input: x_1, x_2, ..., x_T (Quellsequenz)

For t = 1 to T:
    h_t = LSTM(x_t, h_{t-1})

Output: c = h_T (Context Vector)
        oder: c = [h_1, h_2, ..., h_T] (alle States)
```

**Funktion:**
- Komprimiert Input in festen Vektor c
- c soll gesamte Semantik enthalten

#### Decoder
```
Input: y_0 = <START>, c

For t = 1 to T':
    h_t = LSTM(y_{t-1}, h_{t-1}, c)
    P(y_t | y_{1:t-1}, c) = Softmax(W · h_t)
    y_t = sample(P(y_t | ...))
    
    Stop wenn y_t = <END>
```

**Teacher Forcing (Training):**
```
Statt y_{t-1} = eigenes Prediction:
Verwende y_{t-1} = Ground Truth aus Training

Vorteil: Stabilere Gradienten
Nachteil: Train/Test Mismatch (Exposure Bias)
```

---

### 7.3 Bottleneck-Problem ⭐⭐⭐ PRÜFUNGSRELEVANT

**Problem:**
```
c = h_T (letzter Hidden State)
dim(c) = d_h (z.B. 512)

Aber: Satz kann 50+ Wörter haben!
→ Information muss in 512 Dimensionen gepresst werden
→ Informationsverlust bei langen Sätzen
```

**Folgen:**
- Performance degradiert mit Satzlänge
- Wichtige Details gehen verloren
- "Memory Overload" im Encoder

**Visualisierung:**
```
Langer Satz: "The European Parliament voted on Thursday to approve 
              the trade agreement between the European Union and 
              Canada after years of negotiations..."

Encoder: Verarbeitet 30+ Wörter
c: 512-dim Vektor

Decoder: Soll daraus Übersetzung generieren
Problem: Zu wenig Information in c!
```

---

## 8. Attention Mechanismus (Bahdanau Attention) ⭐⭐⭐ PRÜFUNGSRELEVANT

### 8.1 Kernidee

**Innovation (Bahdanau et al., 2014):**
- Decoder sieht **alle** Encoder-Hidden-States
- Nicht nur letzten State h_T
- **Selektiver Fokus** auf relevante Input-Positionen

**Intuition:**
- Beim Übersetzen von Wort y_t:
  - Schaue auf relevante Input-Wörter x_i
  - Ignoriere irrelevante Wörter
- Beispiel: "The cat sat on the mat" → "Die Katze saß auf der Matte"
  - Für "Katze": Fokus auf "cat"
  - Für "Matte": Fokus auf "mat"

---

### 8.2 Bahdanau Attention (Additive Attention) ⭐ AUSWENDIG LERNEN

#### Schritt 1: Alignment Score berechnen

**Für jeden Decoder-Schritt t:**
```
score(s_{t-1}, h_i) = v_a^T · tanh(W_a · s_{t-1} + U_a · h_i)

wobei:
- s_{t-1}: Decoder Hidden State (vorheriger Schritt)
- h_i: Encoder Hidden State an Position i
- W_a, U_a, v_a: Lernbare Parameter
```

**Alternative Scores:**
```
Dot Product: score(s, h) = s^T · h
Scaled Dot Product: score(s, h) = s^T · h / √d  (Transformer!)
General: score(s, h) = s^T · W · h
```

#### Schritt 2: Attention-Gewichte (α)

```
α_{t,i} = softmax(score(s_{t-1}, h_i))
        = exp(score(s_{t-1}, h_i)) / Σ_j exp(score(s_{t-1}, h_j))
```

**Eigenschaften:**
- α_{t,i} ∈ (0, 1)
- Σ_i α_{t,i} = 1 (normalisiert)
- α_{t,i} = "Wie wichtig ist Input-Position i für Output-Position t?"

#### Schritt 3: Context Vector ⭐⭐⭐ PRÜFUNGSRELEVANT

```
c_t = Σ_i α_{t,i} · h_i
```

**Wichtig:**
- c_t ist **unterschiedlich für jeden Decoder-Schritt t!** ⭐
- Bei t=1: Anderer Fokus als bei t=5
- **Dynamische Gewichtung!**

---

### 8.3 Decoder mit Attention

**Update-Gleichung:**
```
s_t = LSTM(y_{t-1}, s_{t-1}, c_t)

wobei:
- c_t = Attention-Weighted Summe der Encoder-States
- s_t = Neuer Decoder-State
```

**Output:**
```
P(y_t | y_{1:t-1}, x) = Softmax(W · [s_t; c_t])
```

**Vollständiger Decoder-Schritt:**
```
1. Berechne Alignment Scores: e_i = score(s_{t-1}, h_i) für alle i
2. Berechne Attention-Gewichte: α_i = softmax(e_i)
3. Berechne Context Vector: c_t = Σ_i α_i · h_i
4. LSTM Update: s_t = LSTM(y_{t-1}, s_{t-1}, c_t)
5. Output: P(y_t) = Softmax(W · [s_t; c_t])
6. Sample oder Argmax: y_t
```

---

### 8.4 Attention Visualisierung ⭐ PRÜFUNGSRELEVANT

**Beispiel Übersetzung DE → EN:**
```
Deutsch:  "Der   Hund  bellt  laut"
          h_1   h_2   h_3    h_4

Englisch: "The   dog   barks  loudly"
          y_1   y_2   y_3     y_4

Attention-Gewichte (α):
         h_1   h_2   h_3   h_4
y_1:    [0.9,  0.05, 0.03, 0.02]  → Fokus auf "Der"
y_2:    [0.02, 0.92, 0.04, 0.02]  → Fokus auf "Hund"
y_3:    [0.01, 0.03, 0.94, 0.02]  → Fokus auf "bellt"
y_4:    [0.02, 0.02, 0.06, 0.90]  → Fokus auf "laut"
```

**Interpretation:**
- Diagonale Struktur → Gutes Alignment
- Zeigt welche Input-Wörter für welche Output-Wörter relevant sind

---

### 8.5 Warum Attention das Bottleneck löst ⭐⭐⭐ PRÜFUNGSRELEVANT

**Ohne Attention:**
```
c = h_T (ein Vektor, feste Größe)
Information: Alle Wörter → ein Vektor → Verlust
```

**Mit Attention:**
```
c_t = Σ_i α_{t,i} · h_i (für jeden Schritt neu)
Information: Alle h_i verfügbar → selektiver Zugriff → kein Verlust!
```

**Vorteile:**
1. **Kein Informationsverlust:** Alle Encoder-States verfügbar
2. **Interpretierbar:** Attention-Gewichte zeigen Alignment
3. **Bessere lange Abhängigkeiten:** Direkter Zugriff auf frühe Positionen
4. **Parallelisierbar:** Alle α_{t,i} gleichzeitig berechenbar

---

## 9. RNN-Parallelisierung ⭐⭐ PRÜFUNGSRELEVANT

### 9.1 Warum RNNs nicht parallelisierbar sind

**Fundamentales Problem:**
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)

Berechnung von h_t benötigt h_{t-1}
Berechnung von h_{t-1} benötigt h_{t-2}
...
```

**Datumabhängigkeitskette:**
```
x_0 → h_0 → h_1 → h_2 → ... → h_T
      ↓     ↓     ↓           ↓
      y_0   y_1   y_2         y_T
```

**Folge:**
- **Strikt sequenzielle Berechnung** über Zeit
- Selbst mit unendlichen GPUs: h_{t+1} kann nicht vor h_t berechnet werden
- **Latency = O(T)** für Sequenzlänge T

---

### 9.2 Was parallelisierbar ist

**Batch-Dimension:**
```
Sequenz 1: x_1^1 → h_1^1 → h_2^1 → ...
Sequenz 2: x_1^2 → h_1^2 → h_2^2 → ...
Sequenz B: x_1^B → h_1^B → h_2^B → ...

→ Alle B Sequenzen parallel berechenbar
→ Aber: Innerhalb jeder Sequenz: sequenziell
```

**Parameter-Updates:**
```
Gradienten über Batch mittelbar
W Update parallel anwendbar
```

**Bidirektionale RNNs:**
```
Vorwärts: h_1^→, h_2^→, ..., h_T^→ (sequenziell)
Rückwärts: h_T^←, h_{T-1}^←, ..., h_1^← (sequenziell)

→ Vorwärts und Rückwärts parallel berechenbar!
→ Aber: Jede Richtung für sich sequenziell
```

---

### 9.3 Vergleich: RNN vs. Transformer

| Eigenschaft | RNN | Transformer |
|-------------|-----|-------------|
| **Zeit-Parallelisierung** | ❌ Nein | ✅ Ja |
| **Batch-Parallelisierung** | ✅ Ja | ✅ Ja |
| **Sequentialität** | O(T) | O(1) |
| **Memory** | O(d) | O(T·d) |
| **Lange Abhängigkeiten** | Schwer (trotz LSTM) | Einfach (Attention) |

**Transformer:**
```
Alle Positionen gleichzeitig berechenbar
Self-Attention: Jede Position sieht alle anderen sofort
→ Massive Parallelisierung auf GPUs
→ Schnelleres Training
```

---

## 10. Suchstrategien während Inferenz ⭐⭐ PRÜFUNGSRELEVANT

### 10.1 Greedy Search

**Algorithmus:**
```
For t = 1 to T:
    P(y_t | y_{1:t-1}) = Model(y_{1:t-1})
    y_t = argmax P(y_t | y_{1:t-1})
```

**Eigenschaften:**
- ✅ Schnell: O(T · |V|)
- ✅ Einfach zu implementieren
- ❌ Nicht optimal: Lokale Maxima ≠ globales Maximum
- ❌ Keine Diversität: Immer gleiche Outputs

**Beispiel:**
```
P("The" | <START>) = 0.4  ← argmax
P("A" | <START>) = 0.3
P("My" | <START>) = 0.2

→ Wähle "The"
→ Aber: Sequenz mit "A" könnte bessere Gesamtwahrscheinlichkeit haben!
```

---

### 10.2 Beam Search ⭐⭐⭐ PRÜFUNGSRELEVANT

**Algorithmus:**
```
Beam Size: k (typisch: 5-10)

Initial: Beams = {<START>}

For t = 1 to T:
    1. Expandiere alle k Beams:
       Für jeden Beam: Berechne P(y_t | Beam)
    
    2. Behalte top-k Sequenzen nach Gesamtwahrscheinlichkeit:
       P(Sequenz) = Π_{t'=1}^{t} P(y_{t'} | y_{1:t'-1})
    
    3. Stop wenn alle Beams <END> oder max Länge erreicht
```

**Beispiel (k=2):**
```
t=1:
  "The" (0.4), "A" (0.3), "My" (0.2)
  → Behalte: "The" (0.4), "A" (0.3)

t=2:
  "The cat" (0.4×0.5=0.20)
  "The dog" (0.4×0.4=0.16)
  "A cat" (0.3×0.6=0.18)
  "A dog" (0.3×0.3=0.09)
  → Behalte: "The cat" (0.20), "A cat" (0.18)

t=3: ...
```

**Eigenschaften:**
- ✅ Bessere Gesamtqualität als Greedy
- ✅ Kontrollierbarer Trade-off (k größer = besser, aber langsamer)
- ❌ Immer noch nicht optimal (nicht alle Pfade)
- ❌ O(k · T · |V|) → langsamer als Greedy

---

### 10.3 Vollständige Suche

**Algorithmus:**
- Alle |V|^T Kombinationen durchprobieren
- Wähle Sequenz mit höchster Wahrscheinlichkeit

**Problem:**
- Zeitkomplexität: O(|V|^T)
- Bei |V|=10.000, T=10: 10.000^10 = 10^40 Kombinationen
- **Nicht praktikabel!**

---

## 11. Lösungen für Vanishing/Exploding Gradients ⭐⭐ PRÜFUNGSRELEVANT

### 11.1 Glorot-Initialisierung (Xavier-Init)

**Idee:**
- Initialisiere Gewichte so, dass Varianz über Layers erhalten bleibt

**Formel:**
```
W ~ N(0, σ²) mit σ² = 2 / (n_in + n_out)

wobei:
- n_in = Anzahl Input-Neuronen
- n_out = Anzahl Output-Neuronen
```

**Effekt:**
- Vermeidet zu große/kleine Gradienten am Anfang
- Stabilisiert Training

---

### 11.2 Aktivierungsfunktionen

**Problem mit Sigmoid:**
```
σ(x) = 1 / (1 + exp(-x))
σ'(x) = σ(x) · (1 - σ(x)) ∈ (0, 0.25]

→ Ableitung immer < 0.25
→ Bei vielen Schritten: Gradient → 0
```

**Besser: Tanh**
```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
tanh'(x) = 1 - tanh²(x) ∈ (0, 1]

→ Ableitung bis 1 (nicht 0.25)
→ Gradient fließt besser
```

**Noch besser: ReLU**
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 wenn x > 0, sonst 0

→ Keine Dämpfung für positive Werte
→ Aber: "Dying ReLU" Problem (Gradient = 0 für x < 0)
```

---

### 11.3 Gradient Clipping ⭐ PRÜFUNGSRELEVANT

**Algorithmus:**
```
g = Gradient
threshold = 1.0 (hyperparameter)

if ||g|| > threshold:
    g = g · (threshold / ||g||)
```

**Effekt:**
- Begrenzt Update-Größe
- Verhindert Exploding Gradients
- Keine Auswirkung auf Vanishing Gradients

---

### 11.4 Batch Normalization

**Idee:**
- Normalisiere Activations über Batch
- Reduziere Internal Covariate Shift

**Formel:**
```
μ = mean(x über Batch)
σ² = var(x über Batch)
x_norm = (x - μ) / √(σ² + ε)
y = γ · x_norm + β  (lernbare Parameter)
```

**Vorteile:**
- Stabilere Gradienten
- Höhere Learning Rates möglich
- Regularisierungseffekt

---

### 11.5 Residuelle Verbindungen

**Idee (ResNet):**
```
y = F(x) + x  (Skip Connection)
```

**Gradient:**
```
∂L/∂x = ∂L/∂y · (∂F/∂x + 1)

→ Der "+1" Term garantiert Gradientenfluss!
→ Selbst wenn ∂F/∂x → 0: Gradient fließt noch
```

**Anwendung in RNNs:**
- Highway Networks
- Skip Connections über mehrere Schritte

---

## 12. Zusammenfassung & Prüfungsrelevanz

### 12.1 Wichtigste Konzepte ⭐⭐⭐

| Konzept | Prüfungsrelevanz | Typische Frage |
|---------|------------------|----------------|
| **BoW/TF-IDF vs. Word2Vec** | 🔴⭐⭐⭐ | Sparse vs. dense, Speicherbedarf |
| **Word2Vec (CBOW/Skip-gram)** | 🔴⭐⭐⭐ | Architektur, Analogien |
| **FastText** | 🟡⭐⭐ | n-grams, OOV-Behandlung |
| **BPE** | 🔴⭐⭐⭐ | Algorithmus, Subword-Tokenisierung |
| **RNN-Formel** | 🔴⭐⭐⭐ | h_t = tanh(W_h·h_{t-1} + W_x·x_t) |
| **Vanishing Gradient** | 🔴⭐⭐⭐ | Ursache, Lösung |
| **LSTM Gates** | 🔴⭐⭐⭐ | 3 Gates, Cell State, Parameter |
| **GRU** | 🟡⭐⭐ | 2 Gates, Unterschied zu LSTM |
| **Seq2Seq** | 🔴⭐⭐⭐ | Encoder-Decoder, Bottleneck |
| **Bahdanau Attention** | 🔴⭐⭐⭐ | Context Vector, α-Gewichte |
| **RNN Parallelisierung** | 🟡⭐⭐ | Warum nicht parallelisierbar? |
| **Beam Search** | 🟡⭐⭐ | Algorithmus, Beam Size |

---

### 12.2 Wichtige Formeln ⭐ AUSWENDIG LERNEN

**TF-IDF:**
```
TF-IDF(w,d,D) = TF(w,d) × log(|D|/|{d∈D: w∈d}|)
```

**RNN:**
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
```

**LSTM:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**LSTM Parameter:**
```
N_params = 4 × d_h × (d_x + d_h + 1)  (mit Bias)
```

**GRU:**
```
z_t = σ(W_z · [h_{t-1}, x_t])
r_t = σ(W_r · [h_{t-1}, x_t])
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Attention:**
```
α_{t,i} = softmax(score(s_{t-1}, h_i))
c_t = Σ_i α_{t,i} · h_i
```

**Cosine Similarity:**
```
cos(v₁, v₂) = (v₁ · v₂) / (|v₁| |v₂|)
```

**Beam Search:**
```
P(Sequenz) = Π_{t'=1}^{t} P(y_{t'} | y_{1:t'-1})
```

---

### 12.3 Typische Prüfungsfragen ⭐⭐⭐

**Frage 1: BoW vs. Word2Vec**
> "Erläutern Sie den Unterschied zwischen BoW/TF-IDF und Word2Vec-Embeddings hinsichtlich Dimension, Sparsity und Eignung für Lernalgorithmen."

**Antwort:**
- BoW/TF-IDF: |V|-dimensional (100.000+), sparse, für lineare Modelle
- Word2Vec: d-dimensional (100-300), dense, für neuronale Netze
- Speicherbedarf: Word2Vec ~300× weniger

**Frage 2: LSTM Parameter**
> "Berechnen Sie die Parameterzahl eines LSTM mit d_x=64 und d_h=128."

**Antwort:**
```
N = 4 × d_h × (d_x + d_h + 1)
  = 4 × 128 × (64 + 128 + 1)
  = 4 × 128 × 193
  = 98.816 Parameter
```

**Frage 3: LSTM Gates**
> "Erklären Sie die Funktion der drei Gates im LSTM."

**Antwort:**
- **Forget Gate:** Was aus Cell State löschen (σ, 0=vergessen, 1=behalten)
- **Input Gate:** Welche neuen Informationen hinzufügen (σ + tanh für Kandidaten)
- **Output Gate:** Was aus Cell State als Output (σ filtert tanh(C_t))

**Frage 4: Seq2Seq Bottleneck**
> "Was ist das Bottleneck-Problem in Seq2Seq und wie löst Attention es?"

**Antwort:**
- Problem: Gesamter Satz in einem Vektor fester Länge (h_T)
- Lösung: Attention gibt Zugriff auf alle Encoder-States
- Context Vector c_t für jeden Decoder-Schritt neu berechnet

**Frage 5: Bahdanau Attention**
> "Ist der Context Vector in Bahdanau-Attention für alle Decoder-Schritte gleich?"

**Antwort:**
- **NEIN!** c_t wird für jeden Schritt t neu berechnet
- α_{t,i} ändert sich mit t (unterschiedlicher Fokus)
- c_t = Σ_i α_{t,i} · h_i (dynamische gewichtete Summe)

**Frage 6: RNN Parallelisierung**
> "Warum können RNNs nicht über die Zeitdimension parallelisiert werden?"

**Antwort:**
- h_t hängt von h_{t-1} ab (Datumabhängigkeitskette)
- Strikt sequenzielle Berechnung erforderlich
- Selbst mit unendlichen Ressourcen: h_{t+1} wartet auf h_t
- Nur Batch-Dimension parallelisierbar

**Frage 7: FastText OOV**
> "Wie behandelt FastText Out-of-Vocabulary-Wörter?"

**Antwort:**
- Zerlege Wort in Character n-grams (Länge 3-6)
- Embedding = Summe der n-gram Embeddings
- Unbekannte Wörter aus bekannten n-grams zusammensetzbar

**Frage 8: BPE Algorithmus**
> "Beschreiben Sie den BPE-Algorithmus zur Subword-Tokenisierung."

**Antwort:**
1. Starte mit allen Characters als Vokabular
2. Zähle alle benachbarten Symbol-Paare im Korpus
3. Finde häufigstes Paar, füge als neues Symbol hinzu
4. Ersetze alle Vorkommen des Paares
5. Wiederhole bis Vokabulargröße erreicht

**Frage 9: Vanishing Gradient**
> "Was verursacht das Vanishing Gradient Problem bei RNNs?"

**Antwort:**
- tanh-Ableitung < 1 (typisch ≈ 0.25)
- Bei Backpropagation über viele Schritte: Produkt vieler Ableitungen → 0
- Gradient für frühe Zeitpunkte verschwindet
- Lösung: LSTM mit additivem Cell State (Constant Error Carousel)

**Frage 10: CBOW vs. Skip-gram**
> "Was ist der Unterschied zwischen CBOW und Skip-gram?"

**Antwort:**
- CBOW: Kontext → Zielwort (schneller, gut für häufige Wörter)
- Skip-gram: Zielwort → Kontext (besser für seltene Wörter, langsamer)

---

## 13. Eigene Notizen & Verständnis

### 13.1 Kernpunkte ⭐

✅ **Word Embeddings:** Dense Vektoren erfassen Semantik durch Kontext
✅ **RNNs:** Verarbeiten Sequenzen, aber sequenzielle Berechnung
✅ **LSTM:** Löst Vanishing Gradient durch Cell State + Gates
✅ **Attention:** Löst Bottleneck durch selektiven Zugriff
✅ **BPE:** Subword-Tokenisierung für kontrolliertes Vokabular

### 13.2 Häufige Fehler ⚠️

❌ LSTM vs. GRU verwechseln (GRU: 2 Gates, kein Cell State)
❌ Attention-Gewichte nicht normalisiert (müssen softmax sein!)
❌ Context Vector als konstant angenommen (ändert sich pro Schritt!)
❌ RNN Parallelisierung falsch eingeschätzt (nur Batch, nicht Zeit)
❌ LSTM Parameter falsch berechnet (4 Gates × (W_x + W_h + b))
❌ BPE mit Word-Level verwechselt (BPE ist Subword!)

### 13.3 Lernstrategie 📚

1. **Formeln auswendig lernen:** LSTM, GRU, Attention, TF-IDF
2. **Parameter berechnen können:** LSTM mit gegebenen d_x, d_h
3. **Konzepte verstehen:** Warum Attention? Warum LSTM?
4. **Vergleiche können:** RNN vs. LSTM, GRU vs. LSTM, BoW vs. Word2Vec
5. **Algorithmen skizzieren:** BPE, Beam Search, Attention

---

**Erstellt:** 2026-03-17 (erweiterte Version)
**Basierend auf:** AdvancedML-02-WordEmbeddings-RNNs.pdf (~36 Seiten) + beispionfragen + alle anderen Quellen
**Umfang:** Vollständige Abdeckung aller PDF-Themen mit Fokus auf klausurrelevante Inhalte
**Geschätzte PDF-Abdeckung:** ~95% (alle Hauptkonzepte abgedeckt)

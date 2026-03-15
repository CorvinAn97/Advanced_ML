# ZUSAMMENFASSUNG 02: Word Embeddings & Recurrent Neural Networks

## Übersicht
- Seitenzahl: ~36 Seiten
- Hauptthemen: Word Embeddings (Word2Vec, FastText), RNN, LSTM, GRU, Seq2Seq, Attention

## Detaillierte Inhalte

### 1. Darstellungsmöglichkeiten von Text

#### Bag of Words (BoW)
- **Idee:** Jedes Wort als One-Hot-Vektor repräsentiert
- **Dimension:** Größe des Vokabulars (oft 30.000-100.000+)
- **Tokenisierung:** Text wird in einzelne Wörter (Tokens) aufgeteilt
- **Beispiel:**
  - Input: "Language technology is awesome"
  - Tokens: Language | technology | is | awesome
  - Vektor: x = [1, 1, 1, 0, 0, 1] (6-dim Vokabular)
- **Problem:** Keine Semantik, alle Wörter gleich weit entfernt

#### TF-IDF (Term Frequency - Inverse Document Frequency)
- **Motivation:** Häufige Wörter ("und", "der") sollten nicht dominieren

**Formeln:**
```
TF(w, d) = (#w in d) / (#Wörter in d)
IDF(w, D) = log(|D| / #{d ∈ D: w ∈ d})
TF-IDF(w, d, D) = TF(w, d) × IDF(w, D)
```
- **TF:** Lokale Wichtigkeit im Dokument
- **IDF:** Globale Seltenheit im Korpus
- **Anwendung:** Dokumentenvergleich (Cosine Similarity), Input für ML-Modelle

**Cosine Similarity:**
```
cos(v₁, v₂) = (v₁ · v₂) / (|v₁| |v₂|)
```

#### Probleme von BoW/TF-IDF
- Reihenfolge der Wörter wird nicht berücksachtigt
- Semantik wird nicht gelernt, sondern manuell als Features eingebaut
- Nützlich für Dokumentenklassifizierung, aber nicht für komplexe NLP

---

### 2. Word Embeddings & Distributed Semantics

#### Motivation
- Ziel: Ähnliche Wörter → Ähnliche Vektoren
- Wörter als Vektoren in ℝ^d (d ≈ 100-300)

#### Distributional Hypothesis
- "You shall know a word by the company it keeps"
- Wörter mit ähnlicher Bedeutung treten in ähnlichen Kontexten auf

#### Word2Vec (2013)
**Zwei Varianten:**
1. **CBOW (Continuous Bag-of-Words):** Sagt zentrales Wort aus Kontextwörtern vorher
2. **Skip-gram:** Sagt Kontextwörter aus zentralem Wort vorher

**Training:**
- Große Textkorpora (Wikipedia, Common Crawl)
- Kontextfenster: ±c Wörter um zentrales Wort
- MLP mit einem Hidden Layer
- Parameter sind die gesuchten Embeddings

**Technisch:**
- Wörter one-hot-encoded
- Input: Mittelung über Kontextwörter (CBOW) oder zentrales Wort (Skip-gram)
- M ∈ ℝ^(V×d): Embedding-Matrix (V = Vokabulargröße, d = Embedding-Dimension)
- Jede Spalte von M = Embedding eines Wortes

**Eigenschaften:**
- Analogien: king - man + woman ≈ queen
- Vektor-Arithmetik erfasst semantische Beziehungen

**Bias in Embeddings:**
- Verstärken gesellschaftliche Vorurteile
- Beispiel: Man : Doctor ≈ Woman : Nurse (geschlechtsspezifisch)

#### FastText (Erweiterung von Word2Vec)
- Wörter als Bag of Character n-grams
- where = \<wh, whe, her, ere, re\>, \<where\>
- Embedding = Summe der n-gram Repräsentationen
- **Vorteile:**
  - Bessere Embeddings für seltene Wörter
  - Out-of-vocabulary Wörter können behandelt werden

#### Character Embeddings
- Einzelne Zeichen als Vektoren
- Wort ist Sequenz aus Char-Embeddings
- Verarbeitung durch CNN/RNN/Pooling
- **Vorteile:**
  - Vollständig OOV-fähig
  - Keine Tokenisierung nötig
  - Robust gegen Rechtschreibfehler

#### Byte Pair Encoding (BPE)
- Subword-Tokenisierungsverfahren
- **Ziel:** Kompromiss zwischen Word-Level und Character-Level
- **Ablauf:**
  1. Initialisiere Vokabular mit allen Zeichen
  2. Füge häufigstes N-Gramm hinzu
  3. Wiederhole bis Vokabulargröße erreicht
- **Ergebnis:** Kontrollierte Vokabulargröße, produktive Morphologie
- WordPiece (Google) = Erweiterung von BPE

---

### 3. Recurrent Neural Networks (RNNs)

#### Warum RNNs?
- Viele Daten sind sequentiell strukturiert
- Reihenfolge trägt semantische Information
- Klassische Netze ignorieren Zeitabhängigkeiten

#### Vanilla RNN Architektur
**Recurrent Connection:**
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
y_t = g(W_y · h_t + b_y)
```

- **h_t:** Hidden State (Gedächtnis)
- **x_t:** Input zum Zeitpunkt t
- **y_t:** Output zum Zeitpunkt t
- **W_h, W_x, W_y:** Gewichtsmatrizen (geteilt über alle Zeitschritte!)

**Eigenschaften:**
- Verarbeitet beliebig lange Sequenzen
- Parameterzahl unabhängig von Inputlänge
- Berechnung sequentiell → schlecht parallelisierbar

#### Backpropagation Through Time (BPTT)
```
Loss: L = Σ_t l(y_t, ŷ_t)
```
- Fehler wird über Zeit rückwärts propagiert
- Parameter sind in jedem Zeitschritt gleich (weight sharing)
- **Problem:** Gradienten über sehr viele Zeitschritte → instabil

#### Truncated BPTT
- Lange Sequenzen werden begrenzt
- Sequenzen manuell kürzen

#### RNN-Probleme
1. **Vanishing Gradient:** Gradienten werden sehr klein → kein Lernen
2. **Exploding Gradient:** Gradienten werden sehr groß → Divergenz
3. **Information Bottleneck:** Hidden State muss alles speichern
4. **Langsame Berechnung:** Keine Parallelisierung möglich

---

### 4. LSTM: Long Short-Term Memory

#### Kernidee
- Expliziter Memory-Mechanismus (Cell State)
- Gates regulieren Informationsfluss
- LSTM kann Information über viele Schritte behalten

#### Komponenten

**Forget Gate:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
- Entscheidet, welche Information aus dem Cell State gelöscht wird
- σ (Sigmoid) ∈ (0, 1): 0 = löschen, 1 = behalten

**Input Gate:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```
- Entscheidet, welche neuen Informationen hinzugefügt werden

**Cell State Update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```
- Alte Information (gefiltert) + Neue Information (gefiltert)

**Output Gate:**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```
- Filtert, welche Teile des Cell State als Output/hidden state dienen

**Zusammenfassung:**
- 3 Gates (f, i, o) + Cell State + Hidden State
- Lernbare Parameter pro Gate: (input_dim + hidden_dim) × hidden_dim + hidden_dim
- Insgesamt: 4 × diese Parameter

#### LSTM-Varianten
- **Peephole Connections:** Gates hängen auch von Cell State ab
- **Coupled Gates:** Forget- und Input-Gate gekoppelt

---

### 5. GRU: Gated Recurrent Unit

#### Kernidee
- Vereinfachte Version von LSTM
- Weniger Parameter, schnelleres Training

#### Komponenten
**Update Gate (z):**
- Kombiniert Forget- und Input-Gate

**Reset Gate (r):**
- Bestimmt, wie viel vom alten Hidden State vergessen wird

**Formeln:**
```
z_t = σ(W_z · [h_{t-1}, x_t])
r_t = σ(W_r · [h_{t-1}, x_t])
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

- **Kein separater Cell State** (nur Hidden State)
- Schneller als LSTM, oft vergleichbare Performance

---

### 6. Bi-directional RNN/LSTM/GRU

#### Idee
- Verarbeite Sequenz von links nach rechts UND von rechts nach links
- Konkateniere Outputs beider Richtungen
- **Vorteil:** Kontext von beiden Seiten verfügbar

---

### 7. Seq2Seq (Encoder-Decoder Architektur)

#### Anwendung
- Machine Translation
- Input: Satz in Sprache A
- Output: Satz in Sprache B

#### Architektur
**Encoder:**
- Verarbeitet Input-Sequenz variabler Länge
- Transformiert in Hidden State Vektor fester Länge
- Letzter Hidden State = Kontext-Vektor

**Decoder:**
- Generiert Output-Sequenz
- Startet mit Hidden State vom Encoder
- Input: Zuvor generiertes Wort (bei t=0: Start-Token)

#### Problem: Bottleneck
- Gesamter Satzinhalt muss in einem Vektor fester Länge gespeichert werden
- Schwierig für lange Sequenzen

---

### 8. Attention Mechanismus (Bahdanau Attention)

#### Kernidee
- Decoder sieht zu jedem Zeitschritt den gesamten Input
- Bewertet Relevanz für aktuellen Zeitschritt

#### Umsetzung
- Decoder hat Zugriff auf alle Hidden States des Encoders
- Berechnet Gewichte α_{t,i} für jeden Encoder-Hidden-State
- **Alignment Model:** a(s_{t-1}, h_i) = Score wie gut Input-Position i und Output-Position t zusammenpassen

**Context Vector:**
```
c_t = Σ_i α_{t,i} · h_i
```

**Gewichte:**
```
α_{t,i} = softmax(score(s_{t-1}, h_i))
```

#### Vorteil
- Deutlich bessere Performance bei langen Sequenzen
- Attention-Visualisierung zeigt "Übersetzungs-Alignment"

---

### 9. Suchstrategien während Inferenz

#### Greedy Search
- Wähle bei jedem Zeitschritt Token mit höchster Wahrscheinlichkeit
- **Eigenschaften:** Schnell, einfach, aber nicht optimal für Gesamtsequenz

#### Beam Search
- Behalte bei jedem Schritt die k wahrscheinlichsten Hypothesen
- Bessere Gesamt-Wahrscheinlichkeit als Greedy
- Kompromiss zwischen Qualität und Rechenzeit

#### Vollständige Suche
- Alle Kombinationen durchprobieren
- Zeitkomplexität: O(V^n) → nicht praktikabel

---

### 10. Lösungen für Vanishing/Exploding Gradients

1. **Glorot-Initialisierung** (Xavier-Init)
2. **Aktivierungsfunktionen:** Keine Sigmoid in Hidden Layers (besser tanh/ReLU)
3. **Batch Normalization**
4. **Gradient Clipping:** Manuelle Begrenzung der Update-Größe
5. **Residuelle Verbindungen:** Skip/Residual/Identity/Highway connections
6. **Spezialisierte Architekturen:** LSTM, GRU

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Word2Vec (CBOW & Skip-gram)
- Warum: Grundlage moderner NLP
- Was: Trainingsprinzip, Embedding-Eigenschaften, Analogien

### ✅ LSTM-Mechanismus
- Warum: Wichtigste RNN-Variante
- Was: Forget/Input/Output Gates, Cell State Update

### ✅ Seq2Seq + Attention
- Warum: Vorläufer der Transformer
- Was: Encoder-Decoder, Bahdanau Attention, Bottleneck-Problem

### ✅ Word Embeddings Eigenschaften
- Warum: Basis für alle nachfolgenden Modelle
- Was: Distributional Hypothesis, Bias, FastText vs Word2Vec

## Formeln/Algorithmen (wichtig)

### TF-IDF
```
TF-IDF(w,d,D) = TF(w,d) × log(|D|/|{d∈D: w∈d}|)
```

### RNN Update
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
```

### LSTM Gates
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output
C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C · [h_{t-1}, x_t])
h_t = o_t ⊙ tanh(C_t)
```

### GRU Update
```
z_t = σ(W_z · [h_{t-1}, x_t])        # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])        # Reset gate
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ tanh(W · [r_t ⊙ h_{t-1}, x_t])
```

### Attention Score
```
α_{t,i} = softmax(a(s_{t-1}, h_i))
c_t = Σ_i α_{t,i} · h_i
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **Word Embeddings:** Dense Vektoren erfassen Semantik durch Kontext
- **RNNs:** Verarbeiten Sequenzen, aber haben Probleme mit langen Abhängigkeiten
- **LSTM:** Löst Vanishing Gradient durch expliziten Memory-Mechanismus
- **Attention:** Löst Bottleneck-Problem von Seq2Seq

### ⚠️ Häufige Fehler
- LSTM vs GRU verwechseln (GRU hat keinen Cell State)
- Attention-Gewichte nicht normalisiert (müssen softmax sein)
- Seq2Seq ohne Attention hat Bottleneck-Problem

### 📝 Prüfungsrelevante Fragen
1. Was ist der Unterschied zwischen CBOW und Skip-gram?
2. Erklären Sie die 3 Gates im LSTM!
3. Was ist das Bottleneck-Problem in Seq2Seq?
4. Wie funktioniert Bahdanau Attention?
5. Was sind die Vor- und Nachteile von GRU vs LSTM?
6. Warum haben RNNs Probleme mit langen Sequenzen?
7. Wie funktioniert FastText?

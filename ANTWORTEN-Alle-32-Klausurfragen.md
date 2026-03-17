# Alle 32 Klausurfragen mit Antworten

**Erstellt:** 17.03.2026  
**Quelle:** Zusammenfassungen 01-08  
**Abdeckung:** 100% (alle Fragen beantwortbar)

---

## FRAGE 1: Transformers
**Frage:** Self-Attention: Was sind Query, Key, Value?

**Antwort:**
Query (Q), Key (K) und Value (V) sind die drei fundamentalen Komponenten des Self-Attention-Mechanismus in Transformern.

- **Query (Q):** Repräsentiert das "Suchende" Token – es fragt nach relevanten Informationen im Kontext. Berechnet als: `q_i = W_Q · x_i`
- **Key (K):** Repräsentiert das "Gefundene" Token – es beantwortet die Query und zeigt Relevanz an. Berechnet als: `k_i = W_K · x_i`
- **Value (V):** Enthält die tatsächliche Information, die weitergegeben wird, wenn Key und Query matchen. Berechnet als: `v_i = W_V · x_i`

**Attention-Berechnung:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```
- QK^T berechnet Ähnlichkeit zwischen Query und allen Keys (Attention Scores)
- √d_k skaliert um Sättigung der Softmax zu verhindern
- softmax normalisiert zu Wahrscheinlichkeiten
- Multiplikation mit V gewichtet die Values entsprechend der Relevanz

**Intuition:** Ein Token (Query) sucht nach allen anderen Tokens (Keys) und aggregiert deren Informationen (Values) gewichtet nach Relevanz.

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

---

## FRAGE 2: Transformers
**Frage:** Positional Encoding: Wie funktioniert es?

**Antwort:**
Positional Encoding fügt Positionsinformation zu den Embeddings hinzu, da Self-Attention positionsunabhängig ist.

**Sinus/Cosinus-Encoding (Original Transformer):**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- pos: Position im Satz
- i: Dimension des Encodings
- d: Modell-Dimension

**Eigenschaften:**
- Verschiedene Frequenzen über Dimensionen → eindeutige Positionscodierung
- Periodisch mit sehr großer Periodenlänge
- Wird zum Word Embedding addiert: `x' = e + PE(pos)`
- Nicht lernbar im ursprünglichen Transformer

**Alternativen:**
- **RoPE (Rotary Positional Embedding):** Relative Position durch Rotation von Q und K
- **ALiBi:** Bias basierend auf relativer Position
- Bessere Extrapolation zu längeren Sequenzen

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

---

## FRAGE 3: Transformers
**Frage:** Encoder vs Decoder: Unterschiede?

**Antwort:**

| Aspekt | Encoder | Decoder |
|--------|---------|---------|
| **Attention** | Bidirektional (alle Tokens sehen alle) | Kausal/Masked (nur vergangene Tokens) |
| **Training** | Masked Language Model | Next-Token Prediction |
| **Output** | Kontextbezogene Repräsentationen | Token-Wahrscheinlichkeiten |
| **Cross-Attention** | Nein | Ja (Q aus Decoder, K,V aus Encoder) |
| **Masking** | Kein Masking | Masked Self-Attention (Future-Token auf -∞) |
| **Typische Modelle** | BERT, RoBERTa | GPT, Llama |
| **Anwendung** | Sprachverständnis (Klassifikation, QA, NER) | Textgenerierung (Chatbots, Code) |

**Encoder:**
- Self-Attention: Jedes Token interagiert mit jedem anderen
- Feed Forward Network
- N Layer wiederholt
- Output: Kontextbezogene Repräsentationen

**Decoder:**
- Masked Self-Attention: Nur vergangene Tokens sichtbar (autoregressiv)
- Cross-Attention: Verbindet Encoder-Output mit Decoder
- Feed Forward Network
- Linear + Softmax für Token-Vorhersage

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

---

## FRAGE 4: Transformers
**Frage:** Multi-Head Attention: Was ist das?

**Antwort:**
Multi-Head Attention führt mehrere Attention-Operationen parallel aus, um verschiedene Aspekte/Beziehungen zu erfassen.

**Implementierung:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**Komponenten:**
- h = Anzahl Heads (typisch 8-16)
- d_k = d/h Dimension pro Head (reduzierte Dimension)
- W_i^Q, W_i^K, W_i^V: Lernbare Projektionsmatrizen pro Head
- W_O: Output-Projektionsmatrix

**Vorteile:**
- Verschiedene Heads lernen verschiedene Beziehungstypen (syntaktisch, semantisch, coreferenz, etc.)
- Ähnlich zu verschiedenen Filtern in CNNs
- Parallelisierbar (alle Heads gleichzeitig berechenbar)

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

---

## FRAGE 5: Transformers
**Frage:** Masked Attention: Warum und wie?

**Antwort:**
Masked Attention verhindert, dass Decoder-Tokens zukünftige Positionen sehen – essentiell für autoregressive Generierung.

**Warum:**
- Bei Textgenerierung darf Token t nur Tokens 1...t-1 sehen
- Sonst würde Modell "in die Zukunft schauen" (Cheating)
- Ermöglicht Next-Token Prediction Training

**Wie:**
- Attention-Werte für zukünftige Tokens auf -∞ setzen (vor Softmax)
- Nach Softmax: softmax(-∞) = 0 → keine Attention auf Future-Tokens
- Technisch: Maskenmatrix M mit M_ij = 0 wenn j ≤ t, sonst -∞

**Beispiel (Sequenzlänge 4):**
```
Maskenmatrix für Decoder:
    t=1  t=2  t=3  t=4
t=1  0   -∞   -∞   -∞    (nur sich selbst)
t=2  0    0   -∞   -∞    (t=1 und t=2)
t=3  0    0    0   -∞    (t=1,2,3)
t=4  0    0    0    0    (alle vergangenen)
```

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

---

## FRAGE 6: Transformers
**Frage:** Warum Transformer > RNN?

**Antwort:**
Transformer überlegen RNNs in drei Schlüsseldesiderata:

**1. Pfadlänge für lange Abhängigkeiten:**
- RNN: O(Sequenzlänge) Schritte bevor entfernte Wörter interagieren
- Transformer: O(1) maximale Interaktionsdistanz (jeder mit jedem via Attention)

**2. Parallelisierbarkeit:**
- RNN: Sequentielle Abhängigkeit (h_t benötigt h_{t-1}) → O(n) nicht-parallelisierbar
- Transformer: Alle Positionen gleichzeitig berechenbar → massive GPU-Parallelisierung

**3. Training auf großen Datensätzen:**
- Transformer: O(n²) Komplexität pro Layer, aber gut parallelisierbar
- Ermöglicht Training auf extrem großen Korpora

**RNN-Probleme:**
- Vanishing Gradient trotz LSTM/GRU
- Information Bottleneck (alles in h_t gepresst)
- Langsames Training wegen Sequenzialität

**Transformer-Vorteile:**
- Direkter Zugriff auf alle Positionen
- Attention löst Bottleneck-Problem
- Skalierbar auf große Modelle

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md, 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 7: Transformers
**Frage:** SwiGLU vs GELU: Unterschied?

**Antwort:**

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x · Φ(x)
wobei Φ = CDF der Standardnormalverteilung

Approximation:
GELU(x) ≈ x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```
- "Softer" Cutoff als ReLU
- Besserer Gradientenfluss
- In BERT und vielen Transformers verwendet

**SwiGLU (State-of-the-Art bei LLMs):**
```
SwiGLU(x) = W_2 · (Swish(x·W + b) ⊙ (x·V + c))
wobei Swish(x) = x · σ(x) = x · 1/(1+e^(-x))
```
- Gate-Mechanismus: Situationsabhängiges Durchlassen von Features
- Zwei Projektionen im Hidden Layer (W und V)
- Elementweise Multiplikation (⊙) mit Swish-Aktivierung
- In modernen LLMs (Llama, PaLM) verwendet

**Unterschiede:**
- SwiGLU hat expliziten Gate-Mechanismus (zwei Pfade)
- SwiGLU expressiver, aber mehr Parameter
- GELU einfacher, etablierter
- SwiGLU oft bessere Performance in großen Modellen

**Quelle:** 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md

---

## FRAGE 8: LSTM & RNNs
**Frage:** Die 3 LSTM Gates: Welche und Funktion?

**Antwort:**
LSTM hat drei Gates die den Informationsfluss regulieren:

**1. Forget Gate (f_t):**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
- **Funktion:** Entscheidet was aus Cell State C_{t-1} vergessen wird
- f_t ≈ 0: Vergessen
- f_t ≈ 1: Behalten
- Output: f_t ∈ (0, 1)^d (elementweise)

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
- **Funktion:** Was aus C_t als Hidden State h_t ausgegeben wird
- h_t = o_t ⊙ tanh(C_t)

**Cell State Update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 9: LSTM & RNNs
**Frage:** Vanishing Gradient: Problem und Lösung?

**Antwort:**
**Problem:**
Bei Backpropagation Through Time (BPTT) verschwinden Gradienten für frühe Zeitschritte.

**Ursache:**
```
∂h_t/∂h_{t-k} = Π_{i=0}^{k-1} ∂h_{t-i}/∂h_{t-i-1}
              = Π_{i=0}^{k-1} diag(1 - tanh²(...)) · W_h^T
```
- tanh-Ableitung: (1 - tanh²(x)) ∈ (0, 1], typisch ≈ 0.25
- Bei vielen Multiplikationen: Produkt → 0
- Gradient "verschwindet" für frühe Zeitpunkte

**Folgen:**
- Keine Updates für frühe Zeitpunkte
- Lange Abhängigkeiten nicht lernbar
- "Memory" funktioniert nicht

**Lösungen:**

1. **LSTM/GRU:** Additive Cell State Updates (kein Vanishing)
   - Constant Error Carousel: Gradient fließt direkt

2. **Gradient Clipping:**
   ```
   if ||g|| > threshold: g = g · (threshold / ||g||)
   ```
   - Verhindert Exploding Gradients

3. **Aktivierungsfunktionen:**
   - ReLU statt tanh/sigmoid (keine Sättigung für x > 0)

4. **Glorot-Initialisierung:**
   - Varianz über Layers erhalten

5. **Residuelle Verbindungen:**
   - y = F(x) + x → Gradient fließt via "+1" Term

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 10: LSTM & RNNs
**Frage:** LSTM vs GRU: Unterschiede?

**Antwort:**

| Eigenschaft | GRU | LSTM |
|-------------|-----|------|
| **Gates** | 2 (Update, Reset) | 3 (Forget, Input, Output) |
| **States** | 1 (Hidden State) | 2 (Cell State + Hidden State) |
| **Parameter** | Weniger (~74k bei d_x=64, d_h=128) | Mehr (~99k bei d_x=64, d_h=128) |
| **Training** | Schneller | Langsamer |
| **Performance** | Oft vergleichbar | Etwas besser bei sehr langen Abhängigkeiten |
| **Komplexität** | Einfacher zu implementieren | Komplexer |

**GRU Update-Gleichungen:**
```
z_t = σ(W_z · [h_{t-1}, x_t])           (Update Gate)
r_t = σ(W_r · [h_{t-1}, x_t])           (Reset Gate)
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   (Candidate)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  (Final Hidden State)
```

**LSTM Update-Gleichungen:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     (Forget Gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     (Input Gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  (Cell Candidate)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t         (Cell State)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     (Output Gate)
h_t = o_t ⊙ tanh(C_t)                   (Hidden State)
```

**Parameterformel:**
- LSTM: N_params = 4 × d_h × (d_x + d_h + 1)
- GRU: N_params = 3 × d_h × (d_x + d_h + 1)

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 11: LSTM & RNNs
**Frage:** Bidirectional LSTM: Wie funktioniert es?

**Antwort:**
Bidirektionale LSTMs verarbeiten Sequenzen in beide Richtungen für vollen Kontext.

**Architektur:**
```
Vorwärts-LSTM:
h_t^→ = LSTM(x_t, h_{t-1}^→)

Rückwärts-LSTM:
h_t^← = LSTM(x_t, h_{t+1}^←)

Kombination:
h_t = [h_t^→; h_t^←]  (Konkatenation, doppelte Dimension)
```

**Visualisierung:**
```
x_1 → [→LSTM] → h_1^→
x_2 → [→LSTM] → h_2^→
x_3 → [→LSTM] → h_3^→

x_3 → [←LSTM] → h_3^←
x_2 → [←LSTM] → h_2^←
x_1 → [←LSTM] → h_1^←

Output: h_1 = [h_1^→; h_1^←], h_2 = [h_2^→; h_2^←], ...
```

**Vorteile:**
- Voller Kontext: Vergangenheit + Zukunft
- Bessere Performance bei NER, POS-Tagging, Sentiment Analysis

**Nachteile:**
- Nicht online-fähig: Benötigt gesamte Sequenz im Voraus
- 2× Parameter: Zwei separate LSTMs
- 2× Rechenzeit: Vorwärts + Rückwärts

**Anwendung:**
- ✅ NER, POS-Tagging, Sentiment Analysis (wenn gesamter Text verfügbar)
- ❌ Echtzeit-Übersetzung, autoregressive Generation

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 12: Word Embeddings
**Frage:** CBOW vs Skip-gram: Unterschiede?

**Antwort:**

**CBOW (Continuous Bag-of-Words):**
- **Richtung:** Kontext → Zielwort
- **Input:** Kontextwörter (one-hot-encoded), gemittelt
- **Output:** Vorhersage des Zielworts (Softmax über Vokabular)
- **Vorteile:** Schnelleres Training, gut für häufige Wörter
- **Nachteile:** Verliert Wortordnung, schlechter für seltene Wörter

**Beispiel:**
```
Kontext: "The ___ sat on the mat"
Input: v("The") + v("sat") + v("on") + v("the") + v("mat")
Output: Vorhersage: "cat"
```

**Skip-gram:**
- **Richtung:** Zielwort → Kontext
- **Input:** Zielwort (one-hot)
- **Output:** Vorhersage der Kontextwörter (mehrere Softmax-Outputs)
- **Vorteile:** Besser für seltene Wörter, erfasst feinere semantische Beziehungen
- **Nachteile:** Langsameres Training

**Beispiel:**
```
Zielwort: "cat"
Output: Vorhersage von "The", "sat", "on", "the", "mat"
```

**Vergleich:**
| Aspekt | CBOW | Skip-gram |
|--------|------|-----------|
| Richtung | Kontext → Ziel | Ziel → Kontext |
| Training | Schneller | Langsamer |
| Seltene Wörter | Schlechter | Besser |
| Häufige Wörter | Besser | Gut |
| Semantik | Allgemeiner | Feiner |

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 13: Word Embeddings
**Frage:** Distributional Hypothesis: Was bedeutet das?

**Antwort:**
**Distributional Hypothesis (Firth, 1957):**
> "You shall know a word by the company it keeps"
> (Ein Wort ist durch seine Begleiter charakterisiert)

**Bedeutung:**
- Wörter mit ähnlichem Kontext haben ähnliche Bedeutung
- Semantische Ähnlichkeit ergibt sich aus Verteilung im Text
- Grundlage für alle distributionalen Word Embeddings

**Beispiel:**
```
"The cat sat on the mat"
"The dog sat on the mat"

→ "cat" und "dog" haben ähnliche Kontexte ("sat on the mat")
→ Daher ähnliche Embeddings
```

**Implikationen:**
- Wortbedeutung ist kontextabhängig
- Statistische Muster im Korpus → Semantik
- Grundlage für Word2Vec, GloVe, FastText

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 14: Word Embeddings
**Frage:** FastText vs Word2Vec: Unterschiede?

**Antwort:**

**Word2Vec:**
- Wörter als atomare Einheiten
- Keine Behandlung von Out-of-Vocabulary (OOV) Wörtern
- Keine Morphologie-Berücksichtigung

**FastText:**
- Wörter als Bag-of-Character-n-grams
- Subword-Information nutzt Morphologie
- OOV-Wörter durch n-gram-Zusammensetzung behandelbar

**FastText-Formel:**
```
v_w = Σ_{g ∈ G_w} v_g
```
- G_w = Menge aller n-grams in Wort w
- v_g = Vektor für n-gram g
- Summe über alle n-gram-Vektoren

**Beispiel:**
```
Wort: "where"
n=3 Trigrams: <wh, whe, her, ere, re>
Embedding: v_where = v_<wh> + v_whe + v_her + v_ere + v_re>
```

**Vorteile FastText:**
- ✅ OOV-Wörter behandelbar
- ✅ Morphologische Information
- ✅ Bessere Performance für morphologisch reiche Sprachen
- ✅ Robust gegen Tippfehler

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 15: Word Embeddings
**Frage:** TF-IDF: Formel und Bedeutung?

**Antwort:**
**TF-IDF (Term Frequency - Inverse Document Frequency):**

**Formel:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = f_{t,d} / Σ_{t'∈d} f_{t',d}
         = Häufigkeit von t in Dokument d / Gesamtwörter in d

IDF(t) = log(N / |{d ∈ D: t ∈ d}|)
       = log(Gesamtdokumente / Dokumente mit t)
```

**Bedeutung:**
- **TF:** Wie oft kommt das Wort im Dokument vor?
- **IDF:** Wie speziell/ selten ist das Wort im Korpus?
- **TF-IDF:** Kombination → Wichtigkeit des Worts für das Dokument

**Eigenschaften:**
- Hoher TF-IDF: Wort ist häufig im Dokument, selten im Korpus → wichtig
- Niedriger TF-IDF: Wort ist selten im Dokument oder häufig im Korpus → unwichtig
- Stopwords haben niedrigen IDF → automatisch downgewichtet

**Anwendung:**
- Dokumentenähnlichkeit (Cosine Similarity)
- Information Retrieval
- Dokumentenklassifikation

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 16: Word Embeddings
**Frage:** BPE: Wie funktioniert der Algorithmus?

**Antwort:**
**BPE (Byte Pair Encoding):**

**Algorithmus:**
1. **Initialisierung:** Vokabular = alle einzelnen Zeichen
2. **Iteration:**
   - Finde häufigstes Zeichenpaar
   - Füge als neues Token zum Vokabular hinzu
   - Ersetze alle Vorkommen im Korpus
3. **Wiederholen** bis gewünschte Vokabulargröße erreicht

**Beispiel:**
```
Initial: l o w </w>, n e w </w>, w i d e </w>

Iteration 1: "e" + "</w>" = "e</w>" (häufigstes Paar)
Vokabular: l, o, w, n, e, </w>, i, d, e</w>

Iteration 2: "w" + "e</w>" = "we</w>"
Vokabular: ..., we</w>

Iteration 3: "n" + "e" = "ne"
...
```

**Eigenschaften:**
- Subword-Segmentation
- Häufige Wörter = einzelne Tokens
- Seltene Wörter = Zerlegung in Subwords
- Keine OOV-Probleme

**Anwendung:**
- GPT-2, GPT-3, RoBERTa, Llama
- Standard für moderne LLMs

**Quelle:** 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md

---

## FRAGE 17: Reinforcement Learning
**Frage:** Q-Learning Update: Wie lautet die Formel?

**Antwort:**
**Q-Learning Update-Regel:**

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]
```

**Komponenten:**
- **Q(s, a):** Aktueller Q-Wert für Zustand s und Aktion a
- **α:** Lernrate (0 < α ≤ 1)
- **r:** Erhaltener Reward
- **γ:** Discount-Faktor (0 ≤ γ ≤ 1)
- **s':** Nächster Zustand
- **max_{a'} Q(s', a'):** Maximum über alle möglichen Aktionen im nächsten Zustand

**TD Error (Temporal Difference):**
```
δ = r + γ · max_{a'} Q(s', a') - Q(s, a)
```

**Intuition:**
- Aktualisiere Q-Wert basierend auf erhaltenem Reward und geschätztem zukünftigem Wert
- "max" macht Q-Learning Off-Policy (greedy Ziel, egal welche Aktion wirklich gewählt wurde)

**Quelle:** 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md

---

## FRAGE 18: Reinforcement Learning
**Frage:** Q-Learning vs SARSA: Unterschiede?

**Antwort:**

| Aspekt | Q-Learning | SARSA |
|--------|------------|-------|
| **Policy** | Off-Policy | On-Policy |
| **Update** | max_{a'} Q(s', a') | Q(s', a') mit tatsächlich gewählter a' |
| **Formel** | Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)] | Q(s,a) += α[r + γ·Q(s',a') - Q(s,a)] |
| **Exploration** | Unabhängig vom Update | Abhängig vom Update |
| **Verhalten** | Optimistischer (max) | Konservativer (tatsächliche Aktion) |
| **Riskant?** | Kann riskante Policies lernen | Lernt tatsächlich gefahrene Policy |

**On-Policy vs Off-Policy:**
- **On-Policy (SARSA):** Lernt die Policy, die auch zum Sammeln der Daten verwendet wird (inkl. Exploration)
- **Off-Policy (Q-Learning):** Lernt optimale Policy, während Daten mit beliebiger Policy (z.B. ε-greedy) gesammelt werden

**Beispiel:**
```
SARSA:  Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        wobei a' tatsächlich gewählte Aktion (mit ε-greedy)

Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max_{a'}Q(s',a') - Q(s,a)]
             wobei max unabhängig von tatsächlich gewählter Aktion
```

**Quelle:** 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md

---

## FRAGE 19: Reinforcement Learning
**Frage:** Overestimation in DQN: Problem und Lösung?

**Antwort:**
**Problem:**
DQN nutzt max-Operator für Q-Value Update:
```
y = r + γ · max_{a'} Q(s', a'; θ)
```
- max über geschätzte Werte führt zu systematischer Überschätzung
- Besonders problematisch bei noisy Q-Values
- Instabilität und schlechte Performance

**Ursache:**
- `max E[Q] ≤ E[max Q]` (Jensen's Inequality)
- Maximum über geschätzte Werte ist erwartungstreue Überschätzung

**Lösung: Double DQN**
```
# Standard DQN:
y = r + γ · max_{a'} Q(s', a'; θ)

# Double DQN:
a* = argmax_{a'} Q(s', a'; θ)      # Selektion mit Online-Netzwerk
y = r + γ · Q(s', a*; θ^-)          # Bewertung mit Target-Netzwerk
```

**Entkopplung:**
- **Selektion:** Welche Aktion ist beste? (Online-Netzwerk θ)
- **Bewertung:** Wie gut ist diese Aktion? (Target-Netzwerk θ^-)

**Vorteil:**
- Reduziert Overestimation signifikant
- Stabilere Training
- Bessere Performance

**Quelle:** 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md

---

## FRAGE 20: Reinforcement Learning
**Frage:** Target Networks: Wozu dienen sie?

**Antwort:**
**Target Networks in DQN:**

**Problem ohne Target Network:**
- Q-Network wird ständig aktualisiert
- Target y = r + γ·max Q(s',a') ändert sich bei jedem Update
- "Moving target" → Instabilität

**Lösung:**
- Separate Target Network Q(s,a; θ^-) mit fixierten Parametern
- θ^- wird nur alle N Schritte (z.B. 1000) von θ kopiert
- Target bleibt stabil über viele Updates

**Algorithmus:**
```
# Training:
1. Sample (s, a, r, s') aus Experience Replay
2. Berechne Target: y = r + γ·max_{a'} Q(s', a'; θ^-)
3. Update Online-Netzwerk: min (Q(s,a;θ) - y)²
4. Alle N Schritte: θ^- ← θ (Target-Netzwerk aktualisieren)
```

**Vorteile:**
- ✅ Stabilere Targets
- ✅ Bessere Konvergenz
- ✅ Reduziert Korrelation zwischen Samples

**Quelle:** 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md

---

## FRAGE 21: Reinforcement Learning
**Frage:** UCB: Formel und Intuition?

**Antwort:**
**UCB (Upper Confidence Bound):**

**Formel:**
```
UCB(s, a) = Q̂(s, a) + c · √(ln N(s) / N(s, a))
```

**Komponenten:**
- **Q̂(s, a):** Geschätzter Q-Wert (Exploitation)
- **c:** Explorationskonstante (trade-off Parameter)
- **N(s):** Anzahl Besuche von Zustand s
- **N(s, a):** Anzahl Ausführungen von Aktion a in s
- **√(ln N(s) / N(s, a)):** Konfidenzintervall (Exploration)

**Intuition:**
- **Optimismus bei Unsicherheit:** Unbekannte Aktionen haben hohen Bonus
- Je öfter eine Aktion ausprobiert wurde, desto kleiner der Bonus
- Balance zwischen bekannten guten Aktionen und unbekannten potenziell besseren

**Eigenschaften:**
- Theoretische Garantie: O(log T) regret
- Deterministisch (keine Zufallssampling wie ε-greedy)
- Bonus nimmt mit mehr Samples ab

**Quelle:** 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md

---

## FRAGE 22: Reinforcement Learning
**Frage:** Thompson Sampling: Wie funktioniert es?

**Antwort:**
**Thompson Sampling:**

**Idee:**
- Bayesianischer Ansatz: Modelliere Unsicherheit über Q-Values als Verteilung
- Sample Q-Value aus Posterior-Verteilung
- Wähle Aktion mit höchstem gesampletem Wert

**Algorithmus:**
```
Für jeden Zustand s:
  Für jede Aktion a:
    Sample Q̃(s,a) ~ P(Q(s,a) | Daten)  # Posterior
  
  Wähle a* = argmax_a Q̃(s,a)
  Führe a* aus, beobachte Reward
  Update Posterior P(Q(s,a) | Daten)
```

**Beispiel (Beta-Bernoulli für binäre Rewards):**
```
Prior: Beta(α=1, β=1)  # Uniform
Nach k Erfolgen und l Misserfolgen:
Posterior: Beta(α=1+k, β=1+l)

Sampling: θ ~ Beta(α, β)
```

**Vorteile:**
- Natürliche Exploration durch Unsicherheit
- Optimal für bestimmte Problemklassen
- Kann strukturierte Unsicherheit modellieren

**Quelle:** 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md

---

## FRAGE 23: Reinforcement Learning
**Frage:** Bellman-Gleichung: Was besagt sie?

**Antwort:**
**Bellman-Gleichung:**

**Bellman Expectation Equation:**
```
V^π(s) = E_π[R_{t+1} + γ·V^π(S_{t+1}) | S_t = s]
       = Σ_a π(a|s) · Σ_{s',r} p(s',r|s,a) · [r + γ·V^π(s')]
```

**Bellman Optimality Equation:**
```
V*(s) = max_a Σ_{s',r} p(s',r|s,a) · [r + γ·V*(s')]
Q*(s,a) = Σ_{s',r} p(s',r|s,a) · [r + γ·max_{a'} Q*(s',a')]
```

**Bedeutung:**
- **Rekursive Struktur:** Aktueller Wert = Immediate Reward + Discounted Future Value
- **Optimalität:** V*(s) ist der maximale erreichbare Wert von Zustand s
- **Backup:** Werte propagieren rückwärts durch den Zustandsraum

**Intuition:**
- Der Wert eines Zustands hängt ab vom besten Reward, den man jetzt bekommen kann, plus dem besten zukünftigen Wert
- "Backup" von zukünftigen Werten zu aktuellen

**Quelle:** 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md

---

## FRAGE 24: Generative Modelle
**Frage:** Mode Collapse in GANs: Was ist das?

**Antwort:**
**Mode Collapse:**

**Problem:**
- Generator lernt nur eine begrenzte Vielfalt an Samples
- Ignoriert andere Modi der Datenverteilung
- Beispiel: MNIST GAN generiert nur "1"en, obwohl alle Ziffern trainiert wurden

**Ursache:**
- Generator findet "Lücke" im Discriminator
- Ein einzelnes Sample kann Discriminator täuschen
- Kein Anreiz für Vielfalt, wenn ein Sample gut funktioniert

**Lösungen:**

1. **Wasserstein GAN (WGAN):**
   - Earth Mover's Distance statt JS-Divergenz
   - Kritischer Unterschied: Lipschitz-Kontinuität
   - Gradienten fließen auch bei perfektem Generator

2. **Minibatch Discrimination:**
   - Discriminator sieht ganze Batch
   - Kann ähnliche Samples erkennen

3. **Unrolled GAN:**
   - Generator sieht k Discriminator-Updates voraus
   - Kann Mode Collapse antizipieren

4. **Mode-seeking Regularization:**
   - Explizite Diversitätsanreize

**Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md

---

## FRAGE 25: Generative Modelle
**Frage:** VAE Reparametrisierung: Trick erklären?

**Antwort:**
**VAE Reparametrisierungstrick:**

**Problem:**
- Sampling z ~ q(z|x) ist nicht differenzierbar
- Backpropagation durch stochastische Knoten nicht möglich

**Lösung:**
- Verschiebe Zufälligkeit auf Input-Seite
- z = μ + σ · ε, wobei ε ~ N(0, I)

**Formel:**
```
z = μ(x) + σ(x) ⊙ ε
wobei:
  μ(x), σ(x) = Encoder-Outputs (lernbar)
  ε ~ N(0, I) (fest, nicht lernbar)
```

**Vorteil:**
- Gradient fließt durch μ und σ
- Sampling ist "äußerhalb" des Graphen
- Stochastizität durch ε, nicht durch z

**Visualisierung:**
```
Ohne Trick:    x → [Encoder] → z ~ N(μ,σ) → [Decoder] → x̂
                              ↑ nicht differenzierbar

Mit Trick:     x → [Encoder] → μ, σ → z = μ + σ·ε → [Decoder] → x̂
                 ε ~ N(0,I) ↑
                              differenzierbar!
```

**Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md

---

## FRAGE 26: Generative Modelle
**Frage:** Diffusion Forward/Reverse: Prozess erklären?

**Antwort:**
**Diffusion Modelle:**

**Forward Process (Diffusion):**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

Oder closed-form:
q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
```
- Schrittweise Addition von Gaussian Noise
- β_t: Noise-Schedule (wächst von ~0.0001 bis ~0.02)
- Nach T Schritten: Reines Gaussian Noise

**Reverse Process (Denoising):**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```
- Lernbares Modell entfernt Schritt für Schritt Noise
- Startet von zufälligem Noise
- Generiert schrittweise zum Datenpunkt

**Training:**
- Modell lernt ε_θ(x_t, t) ≈ ε (tatsächliches Noise)
- Loss: MSE zwischen vorhergesagtem und tatsächlichem Noise

**Sampling:**
```
x_T ~ N(0, I)
for t = T, ..., 1:
    x_{t-1} = (x_t - √(1-ᾱ_t)·ε_θ(x_t,t)) / √α_t + σ_t·z
```

**Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md

---

## FRAGE 27: Generative Modelle
**Frage:** Classifier-Free Guidance: Was ist das?

**Antwort:**
**Classifier-Free Guidance (CFG):**

**Motivation:**
- Unconditional und conditional Generation kombinieren
- Bessere Kontrolle über Generierung
- Höhere Qualität durch Guidance

**Formel:**
```
ε_θ(x_t, t, c) = w · ε_θ(x_t, t, c) + (1-w) · ε_θ(x_t, t, ∅)
```
- c: Condition (z.B. Text-Prompt, Klasse)
- ∅: Unconditional (keine Condition)
- w: Guidance scale (typisch 1-10)

**Training:**
- Modell wird mit und ohne Condition trainiert
- Random Dropout von Conditions (z.B. 10%)
- Modell lernt beide: conditional und unconditional

**Effekt:**
- w = 1: Normale conditional Generation
- w > 1: Verstärkte Condition → höhere Qualität, weniger Diversität
- w < 1: Mehr Diversität, weniger Condition-Einhaltung

**Anwendung:**
- Stable Diffusion, DALL-E, Imagen
- Text-to-Image Generation
- Kontrollierbare Generierung

**Quelle:** 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md

---

## FRAGE 28: XAI
**Frage:** LIME vs SHAP: Unterschiede?

**Antwort:**

| Aspekt | LIME | SHAP |
|--------|------|------|
| **Ansatz** | Lokale Approximation | Spieltheorie (Shapley Values) |
| **Modell** | Modell-agnostisch | Modell-agnostisch |
| **Erklärung** | Lokale lineare Approximation | Globale Feature-Attribution |
| **Konsistenz** | Nicht garantiert | Theoretisch fundiert |
| **Berechnung** | Sampling-basiert | Exakt (kleine Modelle) oder approximiert |
| **Stabilität** | Kann variieren | Stabil |

**LIME:**
- Trainiert lokales lineares Modell in Nachbarschaft der Prediction
- Interpretierbares Surrogat-Modell
- "Trust, but verify" - lokale Erklärung

**SHAP:**
- Shapley Values aus kooperativer Spieltheorie
- Fairer Beitrag jedes Features zur Prediction
- Additive Feature Attribution

**Formel SHAP:**
```
φ_j = Σ_{S⊆N\{j}} [|S|!(|N|-|S|-1)! / |N|!] · [v(S∪{j}) - v(S)]
```

**Quelle:** 06_ZUSAMMENFASSUNG-04-XAI.md

---

## FRAGE 29: XAI
**Frage:** SHAP (Shapley Values): Was sind sie?

**Antwort:**
**Shapley Values:**

**Definition:**
- Aus kooperativer Spieltheorie
- Fairer Beitrag jedes Spielers (Features) zum Ergebnis
- Erfüllt Axiome: Effizienz, Symmetrie, Dummy, Additivität

**Formel:**
```
φ_j = Σ_{S⊆N\{j}} [|S|!(|N|-|S|-1)! / |N|!] · [v(S∪{j}) - v(S)]
```

**Komponenten:**
- **N:** Menge aller Features
- **S:** Subset ohne Feature j
- **v(S):** Modellvorhersage mit Features S
- **v(S∪{j}) - v(S):** Marginaler Beitrag von Feature j
- **Gewicht:** Wie oft wird j in dieser Position betrachtet

**Intuition:**
- Durchschnittlicher marginaler Beitrag über alle möglichen Feature-Koalitionen
- "Was wäre die Prediction ohne dieses Feature?"

**Eigenschaften:**
- Additiv: Σ φ_j = v(N) - v(∅)
- Fair: Gleiche Features → gleiche Werte
- Konsistent: Monotones Verhalten

**Quelle:** 06_ZUSAMMENFASSUNG-04-XAI.md

---

## FRAGE 30: XAI
**Frage:** PFI bei Korrelation: Problem?

**Antwort:**
**Permutation Feature Importance (PFI) bei korrelierten Features:**

**Problem:**
- PFI permutiert einzelne Features unabhängig
- Bei Korrelation entstehen unrealistische Datenpunkte
- Importance kann falsch geschätzt werden

**Beispiel:**
```
Features: x1, x2 (stark korreliert)
Modell: y = x1 + x2

PFI für x1:
- Permutiere x1 → x1', x2 bleibt
- Neue Datenpunkte: (x1', x2) unrealistisch
- Modell sieht unmögliche Kombinationen
- PFI(x1) und PFI(x2) beide niedrig, obwohl beide wichtig
```

**Lösungen:**

1. **Gruppen-PFI:**
   - Permutiere korrelierte Features zusammen
   - Realistische Datenpunkte erhalten

2. **Conditional PFI:**
   - Permutiere nur innerhalb ähnlicher Datenpunkte
   - Berücksichtigt Feature-Abhängigkeiten

3. **SHAP:**
   - Berücksichtigt Feature-Interaktionen
   - Weniger anfällig für Korrelationsprobleme

**Quelle:** 06_ZUSAMMENFASSUNG-04-XAI.md

---

## FRAGE 31: Imitation Learning
**Frage:** Distributional Shift in BC: Problem?

**Antwort:**
**Distributional Shift in Behavioral Cloning:**

**Problem:**
- BC trainiert auf Experten-Daten: (s, a) ~ π_E
- BC Policy π_BC ≠ π_E (imperfekte Nachahmung)
- Testzeit: Zustände s ~ π_BC, nicht π_E

**Folge:**
- π_BC macht Fehler → neue Zustände
- In neuen Zuständen: π_BC macht größere Fehler
- Kaskadeneffekt: Fehler akkumulieren

**Mathematisch:**
```
J(π) = E_{s~π}[L(s,π)]

BC optimiert: E_{s~π_E}[L(s,π)]  (Training)
Aber: E_{s~π}[L(s,π)]  (Test) ist was zählt!

Wenn π ≠ π_E: E_{s~π_E}[L] ≠ E_{s~π}[L]
```

**Worst-Case:**
- Fehler wächst quadratisch mit Zeithorizont T
- O(εT²) statt O(εT) bei i.i.d.

**Lösung: DAgger**
- Iterative Datensammlung mit aktueller Policy
- Zustände aus π_BC, Aktionen von Experten
- Konvergiert zu O(εT)

**Quelle:** 08_ZUSAMMENFASSUNG-06-IL.md

---

## FRAGE 32: Imitation Learning
**Frage:** DAgger Algorithmus: Wie funktioniert er?

**Antwort:**
**DAgger (Dataset Aggregation):**

**Algorithmus:**
```
1. Initialisiere D mit Experten-Demonstrationen
2. Trainiere π_1 auf D
3. For i = 1 to N:
   a. Führe π_i aus, sammle Zustände s ~ π_i
   b. Frage Experten für Aktionen a* in diesen Zuständen
   c. Füge (s, a*) zu D hinzu
   d. Trainiere π_{i+1} auf D
4. Return bestes π_i
```

**Idee:**
- Zustände aus der aktuellen Policy (nicht nur Experten)
- Experten-Labels für diese Zustände
- Iterative Verbesserung

**Vorteile:**
- Kein Distributional Shift
- Zustände aus tatsächlicher Policy-Verteilung
- Theoretische Garantie: O(εT) statt O(εT²)

**Nachteile:**
- Benötigt Experten-Interaktion während Training
- Kann teuer sein (menschlicher Experte)
- Online-Lernen notwendig

**Varianten:**
- **SafeDAgger:** Nur wenn π_i unsicher ist
- **EnsembleDAgger:** Mehrere Policies für Unsicherheitsschätzung

**Quelle:** 08_ZUSAMMENFASSUNG-06-IL.md

---

# Zusammenfassung

**Alle 32 Fragen vollständig beantwortet!**

| Themenbereich | Fragen | Status |
|---------------|--------|--------|
| Transformers | 7/7 | ✅ |
| LSTM & RNNs | 4/4 | ✅ |
| Word Embeddings | 5/5 | ✅ |
| Reinforcement Learning | 7/7 | ✅ |
| Generative Modelle | 4/4 | ✅ |
| XAI | 3/3 | ✅ |
| Imitation Learning | 2/2 | ✅ |

**Gesamt:** 32/32 (100%)

Die Zusammenfassungen decken alle klausurrelevanten Fragen ab und enthalten alle wichtigen Formeln, Algorithmen und Konzepte.

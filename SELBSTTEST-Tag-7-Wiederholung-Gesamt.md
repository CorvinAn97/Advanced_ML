# Selbsttest Tag 7: Gesamtwiederholung Aller Themen

**Umfang:** Umfassender Wiederholungstest zu allen 8 Lernplan-Tagen  
**Zeitansatz:** 60-80 Minuten  
**Hinweis:** Dies ist der finale Test vor der Klausur - alle Themen gemischt!

---

## Teil A: Transformers & Embeddings (10 Fragen)

### Frage 1
**Was sind Query, Key und Value im Self-Attention Mechanismus? Erklären Sie die Analogie zu einer Datenbanksuche.**

<details>
<summary>Antwort anzeigen</summary>

**Query (Q):** Repräsentiert das "Suchende" Token – es fragt nach relevanten Informationen im Kontext. Berechnet als: `q_i = W_Q · x_i`

**Key (K):** Repräsentiert das "Gefundene" Token – es beantwortet die Query und zeigt Relevanz an. Berechnet als: `k_i = W_K · x_i`

**Value (V):** Enthält die tatsächliche Information, die weitergegeben wird, wenn Key und Query matchen. Berechnet als: `v_i = W_V · x_i`

**Attention-Berechnung:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Analogie:** Wie in einer Datenbank sucht der Query nach passenden Keys. Die Ähnlichkeit zwischen Query und Key bestimmt, welche Values (Datensätze) zurückgegeben werden.

</details>

---

### Frage 2
**Warum wird in der Attention-Formel durch √d_k dividiert? Was wäre die Konsequenz ohne diese Skalierung?**

<details>
<summary>Antwort anzeigen</summary>

**Skalierungsfaktor:** 1/√d_k

**Warum:** Die Dot-Products QK^T haben Varianz proportional zu d_k. Bei großen Dimensionen werden die Werte sehr groß.

**Konsequenz ohne Skalierung:**
- Softmax-Eingaben werden extrem groß
- Softmax wird "scharf" (nahezu one-hot)
- Gradienten verschwinden (Softmax-Sättigung)
- Instabiles Training

**Formel:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

</details>

---

### Frage 3
**Vergleichen Sie Encoder-Only (BERT) und Decoder-Only (GPT) Modelle. Nennen Sie jeweils 2 typische Anwendungen.**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | Encoder-Only (BERT) | Decoder-Only (GPT) |
|--------|---------------------|-------------------|
| **Attention** | Bidirektional (alle Tokens sehen alle) | Kausal/Masked (nur vergangene Tokens) |
| **Training** | Masked Language Model | Next-Token Prediction |
| **Anwendungen** | Klassifikation, NER, QA, Sentiment | Textgenerierung, Chatbots, Code |

**Encoder-Only Anwendungen:**
- Named Entity Recognition (NER)
- Sentiment Analysis
- Question Answering
- Textklassifikation

**Decoder-Only Anwendungen:**
- Textgenerierung
- Code-Generierung
- Dialogsysteme
- Autoregressive Übersetzung

</details>

---

### Frage 4
**Was ist das Distributional Hypothesis? Erklären Sie mit einem konkreten Beispiel.**

<details>
<summary>Antwort anzeigen</summary>

**Distributional Hypothesis (Firth, 1957):**
> "You shall know a word by the company it keeps"
> (Ein Wort ist durch seine Begleiter charakterisiert)

**Bedeutung:**
- Wörter mit ähnlichem Kontext haben ähnliche Bedeutung
- Semantische Ähnlichkeit ergibt sich aus Verteilung im Text
- Grundlage für Word2Vec, GloVe, FastText

**Beispiel:**
```
"The cat sat on the mat"
"The dog sat on the mat"

→ "cat" und "dog" haben ähnliche Kontexte ("sat on the mat")
→ Daher ähnliche Embeddings
```

</details>

---

### Frage 5
**Vergleichen Sie CBOW und Skip-gram bei Word2Vec. Wann verwendet man welches?**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | CBOW | Skip-gram |
|--------|------|-----------|
| **Richtung** | Kontext → Zielwort | Zielwort → Kontext |
| **Input** | Mehrere Kontextwörter (gemittelt) | Einzelnes Zielwort |
| **Output** | Vorhersage Zielwort | Vorhersage Kontextwörter |
| **Training** | Schneller | Langsamer |
| **Seltene Wörter** | Schlechter | Besser |
| **Häufige Wörter** | Besser | Gut |

**Wann CBOW:** Große Korpora, häufige Wörter wichtig, schnelles Training
**Wann Skip-gram:** Seltene Wörter wichtig, feine semantische Nuancen

</details>

---

### Frage 6
**Wie funktioniert FastText im Vergleich zu Word2Vec? Wie behandelt es OOV-Wörter?**

<details>
<summary>Antwort anzeigen</summary>

**Word2Vec:**
- Wörter als atomare Einheiten
- Keine OOV-Behandlung
- Keine Morphologie

**FastText:**
- Wörter als Bag-of-Character-n-grams
- Subword-Information nutzt Morphologie

**FastText-Formel:**
```
v_w = Σ_{g ∈ G_w} v_g
```

**Beispiel:**
```
Wort: "where"
n=3 Trigrams: <wh, whe, her, ere, re>
Embedding: v_where = v_<wh> + v_whe + v_her + v_ere + v_re>
```

**OOV-Handling:** Unbekannte Wörter werden aus ihren n-grams komponiert.

</details>

---

### Frage 7
**Was ist Multi-Head Attention und warum ist es besser als ein einzelner Attention-Head?**

<details>
<summary>Antwort anzeigen</summary>

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**Vorteile:**
- Verschiedene Heads lernen verschiedene Beziehungstypen:
  - Syntaktische Beziehungen (Subjekt-Verb)
  - Koreferenz (Pronomen → Nomen)
  - Semantische Ähnlichkeit
  - Positionale Abhängigkeiten
- Parallelisierbar
- Ähnlich zu verschiedenen Filtern in CNNs

**Parameter:**
- h = Anzahl Heads (typisch 8-16)
- d_k = d/h Dimension pro Head

</details>

---

### Frage 8
**Was ist Masked Attention (Causal Masking) und warum wird sie im Decoder benötigt?**

<details>
<summary>Antwort anzeigen</summary>

**Masked Attention:** Verhindert, dass Decoder-Tokens zukünftige Positionen sehen.

**Warum:**
- Bei Textgenerierung darf Token t nur Tokens 1...t-1 sehen
- Sonst würde Modell "in die Zukunft schauen" (Cheating)
- Ermöglicht autoregressive Next-Token Prediction

**Implementierung:**
- Attention-Werte für zukünftige Tokens auf -∞ setzen
- Nach Softmax: softmax(-∞) = 0

**Maskenmatrix (Sequenzlänge 4):**
```
    t=1  t=2  t=3  t=4
t=1  0   -∞   -∞   -∞
t=2  0    0   -∞   -∞
t=3  0    0    0   -∞
t=4  0    0    0    0
```

</details>

---

### Frage 9
**Was ist BPE (Byte Pair Encoding) und wie funktioniert der Algorithmus?**

<details>
<summary>Antwort anzeigen</summary>

**BPE-Algorithmus:**
1. **Initialisierung:** Vokabular = alle einzelnen Zeichen
2. **Iteration:**
   - Finde häufigstes Zeichenpaar
   - Füge als neues Token zum Vokabular hinzu
   - Ersetze alle Vorkommen im Korpus
3. **Wiederholen** bis gewünschte Vokabulargröße erreicht

**Beispiel:**
```
Initial: l o w </w>, n e w </w>, w i d e </w>

Iteration 1: "e" + "</w>" = "e</w>"
Iteration 2: "w" + "e</w>" = "we</w>"
Iteration 3: "n" + "e" = "ne"
```

**Eigenschaften:**
- Subword-Segmentation
- Häufige Wörter = einzelne Tokens
- Seltene Wörter = Zerlegung in Subwords
- Keine OOV-Probleme

</details>

---

### Frage 10
**Warum sind Transformer RNNs/LSTMs überlegen? Nennen Sie 3 Gründe.**

<details>
<summary>Antwort anzeigen</summary>

**1. Pfadlänge für lange Abhängigkeiten:**
- RNN: O(Sequenzlänge) Schritte
- Transformer: O(1) maximale Interaktionsdistanz

**2. Parallelisierbarkeit:**
- RNN: Sequentielle Abhängigkeit (h_t benötigt h_{t-1})
- Transformer: Alle Positionen gleichzeitig berechenbar

**3. Training auf großen Datensätzen:**
- Transformer: O(n²) Komplexität, aber gut parallelisierbar
- Ermöglicht Training auf extrem großen Korpora

**RNN-Probleme:**
- Vanishing Gradient trotz LSTM/GRU
- Information Bottleneck
- Langsames Training wegen Sequenzialität

</details>

---

## Teil B: RNNs, LSTM, Seq2Seq (10 Fragen)

### Frage 11
**Schreiben Sie alle 6 LSTM-Gleichungen aus dem Gedächtnis auf und erklären Sie die Funktion jedes Gates.**

<details>
<summary>Antwort anzeigen</summary>

**LSTM-Gleichungen:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     (Forget Gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     (Input Gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  (Cell Candidate)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t         (Cell State)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     (Output Gate)
h_t = o_t ⊙ tanh(C_t)                   (Hidden State)
```

**Gate-Funktionen:**
- **Forget Gate:** Entscheidet was aus Cell State gelöscht wird (0=vergessen, 1=behalten)
- **Input Gate:** Welche neuen Informationen hinzufügen
- **Output Gate:** Was aus Cell State als Hidden State ausgegeben wird

**Cell State Update:** Additive Struktur ermöglicht direkten Gradientenfluss → kein Vanishing Gradient

</details>

---

### Frage 12
**Was ist das Vanishing Gradient Problem bei RNNs? Wie löst LSTM es?**

<details>
<summary>Antwort anzeigen</summary>

**Problem:**
Bei Backpropagation Through Time (BPTT) verschwinden Gradienten für frühe Zeitschritte.

**Ursache:**
```
∂h_t/∂h_{t-k} = Π_{i=0}^{k-1} diag(1 - tanh²(...)) · W_h^T
```
- tanh-Ableitung ∈ (0, 1], typisch ≈ 0.25
- Produkt vieler kleiner Zahlen → 0

**LSTM-Lösung:**
- **Constant Error Carousel:** Additive Cell State Updates
- Wenn f_t ≈ 1 und i_t ≈ 0: Gradient fließt unverändert durch
- Keine multiplikative Ableitung bei jedem Schritt

**Formel:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

</details>

---

### Frage 13
**Vergleichen Sie LSTM und GRU. Wann verwendet man welches?**

<details>
<summary>Antwort anzeigen</summary>

| Eigenschaft | GRU | LSTM |
|-------------|-----|------|
| **Gates** | 2 (Update, Reset) | 3 (Forget, Input, Output) |
| **States** | 1 (Hidden State) | 2 (Cell + Hidden State) |
| **Parameter** | Weniger (~75%) | Mehr (~100%) |
| **Training** | Schneller | Langsamer |
| **Performance** | Oft vergleichbar | Etwas besser bei langen Abhängigkeiten |

**GRU-Gleichungen:**
```
z_t = σ(W_z · [h_{t-1}, x_t])           (Update Gate)
r_t = σ(W_r · [h_{t-1}, x_t])           (Reset Gate)
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])   (Candidate)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  (Final Hidden State)
```

**Wann LSTM:** Sehr lange Abhängigkeiten, maximale Performance
**Wann GRU:** Schnelleres Training, weniger Parameter, oft ausreichend

</details>

---

### Frage 14
**Wie funktioniert ein bidirektionales LSTM? Was sind Vor- und Nachteile?**

<details>
<summary>Antwort anzeigen</summary>

**Architektur:**
```
Vorwärts-LSTM:  h_t^→ = LSTM(x_t, h_{t-1}^→)
Rückwärts-LSTM: h_t^← = LSTM(x_t, h_{t+1}^←)
Kombination:   h_t = [h_t^→; h_t^←]
```

**Vorteile:**
- Voller Kontext: Vergangenheit + Zukunft
- Bessere Performance bei NER, POS-Tagging, Sentiment Analysis

**Nachteile:**
- Nicht online-fähig: Benötigt gesamte Sequenz im Voraus
- 2× Parameter: Zwei separate LSTMs
- 2× Rechenzeit: Vorwärts + Rückwärts

**Anwendung:**
- ✅ NER, POS-Tagging, Sentiment Analysis
- ❌ Echtzeit-Übersetzung, autoregressive Generation

</details>

---

### Frage 15
**Berechnen Sie die Parameterzahl eines LSTM mit d_x=64 und d_h=128. Zeigen Sie Ihre Rechnung.**

<details>
<summary>Antwort anzeigen</summary>

**LSTM Parameterformel:**
```
N = 4 × d_h × (d_x + d_h + 1)
```

**Berechnung:**
```
N = 4 × 128 × (64 + 128 + 1)
N = 4 × 128 × 193
N = 512 × 193
N = 98.816 Parameter
```

**Aufschlüsselung pro Gate:**
- Input-Gewicht: 64 × 128 = 8.192
- Hidden-Gewicht: 128 × 128 = 16.384
- Bias: 128
- Pro Gate: 8.192 + 16.384 + 128 = 24.704
- Gesamt: 4 × 24.704 = 98.816

</details>

---

### Frage 16
**Was ist das Bottleneck-Problem in Seq2Seq ohne Attention?**

<details>
<summary>Antwort anzeigen</summary>

**Problem:**
- Gesamter Input-Satz muss in einen einzigen Vektor c = h_T komprimiert werden
- Bei langen Sätzen: Informationsverlust
- c hat feste Dimension (z.B. 512), unabhängig von Satzlänge

**Folge:**
- Encoder-Output h_T muss alle Informationen des Satzes kodieren
- Bei langen Sätzen: Wichtige Informationen gehen verloren
- Decoder hat nur Zugriff auf diesen einen Vektor

**Lösung durch Attention:**
- Decoder hat Zugriff auf alle Encoder-States h_i
- Selektiver Fokus durch Attention-Gewichte α_{t,i}
- Kein Informationsverlust

</details>

---

### Frage 17
**Erklären Sie Bahdanau Attention (Additive Attention). Wie berechnet sich der Context Vector?**

<details>
<summary>Antwort anzeigen</summary>

**Bahdanau Attention (3 Schritte):**

**1. Alignment Score:**
```
e_{t,i} = score(s_{t-1}, h_i) = v_a^T · tanh(W_s · s_{t-1} + W_h · h_i)
```

**2. Attention-Gewichte:**
```
α_{t,i} = softmax(e_{t,i}) = exp(e_{t,i}) / Σ_j exp(e_{t,j})
```

**3. Context Vector:**
```
c_t = Σ_i α_{t,i} · h_i
```

**Eigenschaften:**
- c_t wird für jeden Decoder-Schritt t neu berechnet
- α_{t,i} ändert sich mit t (unterschiedlicher Fokus)
- Alignment Score verwendet Feed-Forward Network

</details>

---

### Frage 18
**Was ist Teacher Forcing beim Training von Seq2Seq-Modellen? Was ist das Exposure Bias Problem?**

<details>
<summary>Antwort anzeigen</summary>

**Teacher Forcing:**
- Beim Training wird das echte nächste Token als Input verwendet
- Nicht die eigene Vorhersage
- Ermöglicht paralleles Training aller Positionen

**Exposure Bias:**
- Modell wird nur mit echten Daten trainiert
- Bei Inferenz: Modell macht Fehler → neue Zustände
- In neuen Zuständen: Größere Fehler (Distributional Shift)
- Fehler akkumulieren sich

**Mathematisch:**
```
Training:  E_{s~π_E}[L(s,π)]
Test:      E_{s~π}[L(s,π)]

Wenn π ≠ π_E: Quadratische Fehlerakkumulation O(εT²)
```

</details>

---

### Frage 19
**Was ist TF-IDF? Berechnen Sie TF-IDF für "cat" in einem Dokument.**

<details>
<summary>Antwort anzeigen</summary>

**TF-IDF Formel:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = f_{t,d} / Σ_{t'∈d} f_{t',d}
IDF(t) = log(N / |{d ∈ D: t ∈ d}|)
```

**Beispiel:**
- D1: "The cat sat on the mat" (6 Wörter)
- D2: "The dog sat on the log" (6 Wörter)
- D3: "The cat and the dog are friends" (8 Wörter)

**Berechnung:**
```
TF("cat", D1) = 1/6 ≈ 0.167
IDF("cat") = log(3/2) ≈ 0.405
TF-IDF = 0.167 × 0.405 ≈ 0.068
```

**Bedeutung:**
- Hoher TF-IDF: Wort ist häufig im Dokument, selten im Korpus
- Niedriger TF-IDF: Wort ist selten im Dokument oder häufig im Korpus

</details>

---

### Frage 20
**Vergleichen Sie RNN, LSTM und Transformer hinsichtlich Pfadlänge, Parallelisierung und Langzeitabhängigkeiten.**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | RNN | LSTM | Transformer |
|--------|-----|------|-------------|
| **Pfadlänge** | O(n) | O(n) | O(1) |
| **Parallelisierung** | Nein | Nein | Ja |
| **Langzeitabhängigkeiten** | Schlecht | Gut | Sehr gut |
| **Komplexität** | O(n) | O(n) | O(n²) |
| **Speicher** | O(1) | O(1) | O(n²) |

**Erklärung:**
- **RNN:** Sequentielle Verarbeitung, Vanishing Gradient
- **LSTM:** Gates ermöglichen Langzeitabhängigkeiten, aber immer noch sequentiell
- **Transformer:** Direkte Verbindungen durch Attention, aber quadratische Komplexität

</details>

---

## Teil C: RL & Q-Learning (10 Fragen)

### Frage 21
**Schreiben Sie die Q-Learning Update-Regel auf und erklären Sie alle Komponenten.**

<details>
<summary>Antwort anzeigen</summary>

**Q-Learning Update:**
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

**TD Error:**
```
δ = r + γ · max_{a'} Q(s', a') - Q(s, a)
```

**Off-Policy:** Lernt optimale Policy unabhängig von der Behavior-Policy (ε-greedy).

</details>

---

### Frage 22
**Was ist der Unterschied zwischen Q-Learning (Off-Policy) und SARSA (On-Policy)?**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | Q-Learning | SARSA |
|--------|------------|-------|
| **Policy** | Off-Policy | On-Policy |
| **Update** | max_{a'} Q(s', a') | Q(s', a') mit tatsächlich gewählter a' |
| **Formel** | Q(s,a) += α[r + γ·max Q(s',a') - Q(s,a)] | Q(s,a) += α[r + γ·Q(s',a') - Q(s,a)] |
| **Exploration** | Unabhängig vom Update | Abhängig vom Update |
| **Verhalten** | Optimistischer (max) | Konservativer (tatsächliche Aktion) |

**Beispiel (Cliff Walking):**
- **Q-Learning:** Lernt optimalen Pfad (entlang Cliff), fällt oft runter (ε-greedy Fehler)
- **SARSA:** Lernt sicheren Pfad (weg vom Cliff), berücksichtigt ε-Fehler

</details>

---

### Frage 23
**Was ist das Overestimation Problem in DQN und wie löst Double DQN es?**

<details>
<summary>Antwort anzeigen</summary>

**Problem:**
- DQN nutzt max-Operator: `y = r + γ · max_{a'} Q(s', a'; θ)`
- max über geschätzte Werte führt zu systematischer Überschätzung
- `max E[Q] ≤ E[max Q]` (Jensen's Inequality)

**Double DQN Lösung:**
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

**Vorteil:** Reduziert Overestimation signifikant

</details>

---

### Frage 24
**Wozu dienen Target Networks in DQN?**

<details>
<summary>Antwort anzeigen</summary>

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
1. Sample (s, a, r, s') aus Experience Replay
2. Berechne Target: y = r + γ·max_{a'} Q(s', a'; θ^-)
3. Update Online-Netzwerk: min (Q(s,a;θ) - y)²
4. Alle N Schritte: θ^- ← θ
```

**Vorteile:**
- Stabilere Targets
- Bessere Konvergenz
- Reduziert Korrelation zwischen Samples

</details>

---

### Frage 25
**Was ist UCB (Upper Confidence Bound)? Erklären Sie die Formel.**

<details>
<summary>Antwort anzeigen</summary>

**UCB Formel:**
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
- "Optimismus bei Unsicherheit"
- Unbekannte Aktionen haben hohen Bonus
- Bonus nimmt mit mehr Samples ab

**Eigenschaften:**
- Theoretische Garantie: O(log T) regret
- Deterministisch (kein Zufallssampling wie ε-greedy)

</details>

---

### Frage 26
**Wie funktioniert Thompson Sampling?**

<details>
<summary>Antwort anzeigen</summary>

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

**Beispiel (Beta-Bernoulli):**
```
Prior: Beta(α=1, β=1)  # Uniform
Nach k Erfolgen und l Misserfolgen:
Posterior: Beta(α=1+k, β=1+l)

Sampling: θ ~ Beta(α, β)
```

**Vorteile:**
- Natürliche Exploration durch Unsicherheit
- Optimal für bestimmte Problemklassen

</details>

---

### Frage 27
**Was besagt die Bellman-Gleichung?**

<details>
<summary>Antwort anzeigen</summary>

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
Der Wert eines Zustands hängt ab vom besten Reward, den man jetzt bekommen kann, plus dem besten zukünftigen Wert.

</details>

---

### Frage 28
**Was ist Experience Replay und warum wird es verwendet?**

<details>
<summary>Antwort anzeigen</summary>

**Experience Replay:**
- Speichere Transitions (s, a, r, s') in Buffer
- Sample Mini-Batches uniform für Training

**Vorteile:**
1. **Dekorrelation:** Sequentielle Samples sind korreliert (Markov-Kette). Uniform-Sampling bricht Korrelation.
2. **Effizienz:** Jede Transition kann mehrfach verwendet werden
3. **Stabilität:** Vermeidet "catastrophic forgetting"

**Implementierung:**
- Buffer-Größe: 10.000 bis 1.000.000 Transitions
- Mini-Batch-Größe: 32 bis 512

**Prioritized Experience Replay:**
- Sample wichtige Transitions häufiger (hoher TD-Error)

</details>

---

### Frage 29
**Was ist der Unterschied zwischen Model-Based und Model-Free RL?**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | Model-Based | Model-Free |
|--------|-------------|------------|
| **Modell** | Lernt Environment-Modell p(s',r\|s,a) | Kein Modell |
| **Planung** | Kann planen (MCTS, Dyna) | Direkte Policy-Optimierung |
| **Sample Efficiency** | Hoch (nutzt Modell) | Niedrig (braucht viele Samples) |
| **Beispiele** | AlphaZero, MuZero, Dyna-Q | Q-Learning, DQN, PPO |

**Model-Based:**
- Lernt Transition Dynamics
- Kann simulieren und planen
- Bessere Sample Efficiency

**Model-Free:**
- Lernt direkt Policy oder Value Function
- Keine Modellierung nötig
- Einfacher, aber mehr Samples nötig

</details>

---

### Frage 30
**Was ist der Unterschied zwischen Value-Based und Policy-Based RL?**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | Value-Based | Policy-Based |
|--------|-------------|--------------|
| **Repräsentation** | Q(s,a) oder V(s) | π(a\|s) direkt |
| **Policy** | Implizit (argmax Q) | Explizit |
| **Exploration** | ε-greedy, UCB | Stochastic Policy |
| **Continuous Actions** | Schwierig | Einfach |
| **Beispiele** | Q-Learning, DQN | REINFORCE, PPO |

**Value-Based:**
- Lernt Value Function
- Policy ist implizit (greedy w.r.t. Q)

**Policy-Based:**
- Lernt Policy direkt
- Kann stochastische Policies lernen
- Geeignet für kontinuierliche Aktionen

</details>

---

## Teil D: Generative Modelle & XAI (10 Fragen)

### Frage 31
**Was ist Mode Collapse in GANs? Nennen Sie Lösungen.**

<details>
<summary>Antwort anzeigen</summary>

**Mode Collapse:**
- Generator lernt nur eine begrenzte Vielfalt an Samples
- Ignoriert andere Modi der Datenverteilung
- Beispiel: MNIST GAN generiert nur "1"en

**Ursache:**
- Generator findet "Lücke" im Discriminator
- Ein einzelnes Sample kann Discriminator täuschen
- Kein Anreiz für Vielfalt

**Lösungen:**

1. **Wasserstein GAN (WGAN):**
   - Earth Mover's Distance statt JS-Divergenz
   - Lipschitz-Kontinuität durch Weight Clipping oder Gradient Penalty

2. **Minibatch Discrimination:**
   - Discriminator sieht ganze Batch
   - Kann ähnliche Samples erkennen

3. **Unrolled GAN:**
   - Generator sieht k Discriminator-Updates voraus

4. **Mode-seeking Regularization:**
   - Explizite Diversitätsanreize

</details>

---

### Frage 32
**Erklären Sie den Reparametrisierungstrick bei VAEs.**

<details>
<summary>Antwort anzeigen</summary>

**Problem:**
- Sampling z ~ q(z|x) ist nicht differenzierbar
- Backpropagation durch stochastische Knoten nicht möglich

**Lösung:**
```
z = μ(x) + σ(x) ⊙ ε
wobei:
  μ(x), σ(x) = Encoder-Outputs (lernbar)
  ε ~ N(0, I) (fest, nicht lernbar)
```

**Vorteil:**
- Gradient fließt durch μ und σ
- Sampling ist "außerhalb" des Graphen
- Stochastizität durch ε, nicht durch z

**Visualisierung:**
```
Ohne Trick:    x → [Encoder] → z ~ N(μ,σ) → [Decoder] → x̂
                              ↑ nicht differenzierbar

Mit Trick:     x → [Encoder] → μ, σ → z = μ + σ·ε → [Decoder] → x̂
                 ε ~ N(0,I) ↑
                              differenzierbar!
```

</details>

---

### Frage 33
**Erklären Sie Forward und Reverse Process bei Diffusion Modellen.**

<details>
<summary>Antwort anzeigen</summary>

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
- Neuronales Netzwerk lernt Noise vorherzusagen
- Startet von zufälligem Noise
- Generiert schrittweise zum Datenpunkt

**Training:**
- Modell lernt ε_θ(x_t, t) ≈ ε (tatsächliches Noise)
- Loss: MSE zwischen vorhergesagtem und tatsächlichem Noise

</details>

---

### Frage 34
**Was ist Classifier-Free Guidance (CFG) bei Diffusion Modellen?**

<details>
<summary>Antwort anzeigen</summary>

**Classifier-Free Guidance:**

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

</details>

---

### Frage 35
**Vergleichen Sie GANs, VAEs und Diffusion Modelle.**

<details>
<summary>Antwort anzeigen</summary>

| Aspekt | GAN | VAE | Diffusion |
|--------|-----|-----|-----------|
| **Training** | Adversarial (Minimax) | ELBO Maximierung | Noise Prediction |
| **Latent Space** | Implizit | Explizit (strukturiert) | Keiner (Markov Chain) |
| **Sample Quality** | Sehr hoch | Mittel | Sehr hoch |
| **Training Stability** | Instabil (Mode Collapse) | Stabil | Stabil |
| **Sampling Speed** | Schnell (1 Forward Pass) | Schnell | Langsam (1000 Schritte) |
| **Likelihood** | Nicht verfügbar | Approximierbar | Verfügbar |

**GAN:**
- Generator + Discriminator
- Hohe Qualität, aber instabil

**VAE:**
- Encoder + Decoder
- Strukturierter Latent Space

**Diffusion:**
- Iteratives Denoising
- Höchste Qualität, aber langsam

</details>

---

### Frage 36
**Was ist LIME und wie funktioniert es?**

<details>
<summary>Antwort anzeigen</summary>

**LIME (Local Interpretable Model-agnostic Explanations):**

**Idee:**
- Approximiere komplexes Modell lokal durch interpretierbares Surrogate-Modell

**Algorithmus:**
1. Wähle zu erklärende Instanz
2. Generiere Perturbationen (kleine Änderungen der Eingabe)
3. Hole Vorhersagen vom Originalmodell für Perturbationen
4. Gewichte: Lokale Gewichtung (nahe Instanz = höheres Gewicht)
5. Trainiere Surrogate (linear): Minimiere gewichteten Fehler
6. Erklärung: Lineare Koeffizienten des Surrogate

**Eigenschaften:**
- **Lokal:** Gültig nur nahe der gewählten Instanz
- **Modellunabhängig:** Black-Box-fähig
- **Surrogate:** Linear, Entscheidungsbaum, etc.

**Nachteil:** Instabil (Perturbationen zufällig), keine theoretische Garantie

</details>

---

### Frage 37
**Was sind SHAP (Shapley Values)? Erklären Sie die Formel.**

<details>
<summary>Antwort anzeigen</summary>

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

</details>

---

### Frage 38
**Was ist das Problem von PFI (Permutation Feature Importance) bei korrelierten Features?**

<details>
<summary>Antwort anzeigen</summary>

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

2. **Conditional PFI:**
   - Permutiere nur innerhalb ähnlicher Datenpunkte

3. **SHAP:**
   - Berücksichtigt Feature-Interaktionen
   - Weniger anfällig für Korrelationsprobleme

</details>

---

### Frage 39
**Vergleichen Sie LIME und SHAP.**

<details>
<summary>Antwort anzeigen</summary>

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
- "Trust, but verify" - lokale Erklärung

**SHAP:**
- Shapley Values aus kooperativer Spieltheorie
- Fairer Beitrag jedes Features zur Prediction
- Additive Feature Attribution

</details>

---

### Frage 40
**Was ist Integrated Gradients und wann wird es verwendet?**

<details>
<summary>Antwort anzeigen</summary>

**Integrated Gradients:**

**Idee:**
- Attributiert Prediction-Änderung zu Features
- Baseline (z.B. schwarzes Bild) zu Input

**Formel:**
```
IG_i(x) = (x_i - x_i') × ∫_0^1 [∂F(x' + α(x - x'))/∂x_i] dα
```

**Eigenschaften:**
- Saxiome: Sensitivity, Implementation Invariance, Completeness
- Geeignet für neuronale Netze
- Berechnet Gradienten entlang des Pfads von Baseline zu Input

**Anwendung:**
- Bildklassifikation (welche Pixel sind wichtig?)
- Textklassifikation (welche Wörter sind wichtig?)
- Wenn Gradienten verfügbar sind

</details>

---

## Teil E: Exploration, IL, RLHF (10 Fragen)

### Frage 41
**Was ist das Distributional Shift Problem in Behavioral Cloning?**

<details>
<summary>Antwort anzeigen</summary>

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

</details>

---

### Frage 42
**Wie funktioniert der DAgger Algorithmus?**

<details>
<summary>Antwort anzeigen</summary>

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
- Online-Lernen notwendig

</details>

---

### Frage 43
**Was ist RLHF (Reinforcement Learning from Human Feedback)?**

<details>
<summary>Antwort anzeigen</summary>

**RLHF:**

**Prozess:**
1. **Supervised Fine-Tuning (SFT):** Trainiere Modell auf Demonstrationen
2. **Reward Model Training:** Trainiere Reward Model auf menschlichen Vergleichen
3. **RL Training:** Optimiere Policy mit PPO gegen Reward Model

**Reward Model:**
- Input: Prompt + Response
- Output: Skalarer Reward
- Trainiert auf menschlichen Vergleichen: "Antwort A ist besser als B"

**PPO Training:**
```
maximize E[RM(x,y) - β·KL(π||π_ref)]
```
- RM: Reward Model
- β·KL: Regularisierung gegen Drift vom SFT-Modell

**Anwendung:**
- ChatGPT, Claude, Llama-2-Chat
- Alignment mit menschlichen Präferenzen

</details>

---

### Frage 44
**Was ist DPO (Direct Preference Optimization) und wie unterscheidet es sich von RLHF?**

<details>
<summary>Antwort anzeigen</summary>

**DPO (Direct Preference Optimization):**

**Idee:**
- Optimiere direkt auf Preference-Daten
- Kein separates Reward Model nötig
- Kein RL-Training (PPO)

**Formel:**
```
L_DPO(π_θ; π_ref) = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Unterschied zu RLHF:**

| Aspekt | RLHF | DPO |
|--------|------|-----|
| **Reward Model** | Explizit trainiert | Nicht nötig |
| **RL Algorithmus** | PPO | Keiner (direkte Optimierung) |
| **Stabilität** | Instabil (PPO) | Stabil |
| **Komplexität** | Hoch | Niedrig |

**Vorteile DPO:**
- Einfacher zu implementieren
- Stabileres Training
- Keine Hyperparameter-Tuning für PPO

</details>

---

### Frage 45
**Was ist ε-greedy Exploration? Was sind die Vor- und Nachteile?**

<details>
<summary>Antwort anzeigen</summary>

**ε-greedy:**

**Algorithmus:**
```
Mit Wahrscheinlichkeit ε:
  Wähle zufällige Aktion (Exploration)
Mit Wahrscheinlichkeit 1-ε:
  Wähle beste Aktion laut Q (Exploitation)
```

**Vorteile:**
- Einfach zu implementieren
- Garantiert Exploration (GLIE wenn ε → 0)
- Funktioniert gut in der Praxis

**Nachteile:**
- Uniforme Exploration (alle Aktionen gleich wahrscheinlich)
- Keine Berücksichtigung von Unsicherheit
- Kann ineffizient sein (exploriert auch unwahrscheinliche Aktionen)

**Varianten:**
- **Decaying ε:** ε nimmt mit Zeit ab
- **Adaptive ε:** ε basierend auf Performance

</details>

---

### Frage 46
**Was ist der Unterschied zwischen Exploration und Exploitation?**

<details>
<summary>Antwort anzeigen</summary>

**Exploration vs Exploitation:**

**Exploration:**
- Ausprobieren neuer Aktionen
- Sammeln von Information
- Langfristig bessere Entscheidungen

**Exploitation:**
- Nutzen bekannter bester Aktionen
- Maximieren des aktuellen Rewards
- Kurzfristig optimal

**Dilemma:**
- Nur Exploration: Sammelt Information, aber kein Reward
- Nur Exploitation: Bleibt bei suboptimalen Lösungen stecken
- Balance nötig für optimales Lernen

**Strategien:**
- ε-greedy: Zufällige Exploration
- UCB: Optimismus bei Unsicherheit
- Thompson Sampling: Bayesianische Exploration

</details>

---

### Frage 47
**Was ist Inverse Reinforcement Learning (IRL)?**

<details>
<summary>Antwort anzeigen</summary>

**Inverse Reinforcement Learning:**

**Problemstellung:**
- Gegeben: Experten-Demonstrationen
- Gesucht: Reward Function die das Verhalten erklärt

**Im Gegensatz zu RL:**
- RL: Reward gegeben, Policy gesucht
- IRL: Demonstrationen gegeben, Reward gesucht

**Algorithmen:**
- **Maximum Margin:** Finde Reward wo Experte optimal ist
- **Maximum Entropy:** Probabilistisches Modell
- ** apprenticeship Learning:** Lerne Policy über gelernte Reward

**Anwendung:**
- Autonomes Fahren (menschliches Fahrverhalten)
- Robotik (imitiere Menschen)
- Wenn Reward schwer zu spezifizieren ist

</details>

---

### Frage 48
**Was ist Causal Confusion in Imitation Learning?**

<details>
<summary>Antwort anzeigen</summary>

**Causal Confusion:**

**Problem:**
- Policy lernt korrelierte Features statt kausale
- Beispiel: Bremslicht vor dem Bremsen
- IL lernt: "Wenn Bremslicht, dann bremsen"
- Aber: Bremslicht ist Konsequenz, nicht Ursache

**Folge:**
- Policy funktioniert im Training
- Fails bei Test (wenn Bremslicht ausfällt)
- Falsche Kausalitäten gelernt

**Lösungen:**
- **Causal Discovery:** Identifiziere kausale Struktur
- **Interventions:** Trainiere mit Interventionen
- **Counterfactual Reasoning:** "Was wäre wenn..."

</details>

---

### Frage 49
**Was ist der Unterschied zwischen On-Policy und Off-Policy RL?**

<details>
<summary>Antwort anzeigen</summary>

**On-Policy vs Off-Policy:**

| Aspekt | On-Policy | Off-Policy |
|--------|-----------|------------|
| **Policy** | Lernt Policy, die auch sammelt | Lernt andere Policy als die, die sammelt |
| **Daten** | Nur aktuelle Policy | Kann alte Daten verwenden |
| **Sample Efficiency** | Niedrig | Hoch |
| **Beispiele** | SARSA, PPO, A3C | Q-Learning, DQN, DDPG |

**On-Policy:**
- Lernt die Policy, die auch zum Sammeln verwendet wird
- Daten von alter Policy können nicht wiederverwendet werden
- Einfacher, aber weniger sample-effizient

**Off-Policy:**
- Lernt optimale Policy, während Daten mit beliebiger Policy gesammelt werden
- Experience Replay möglich
- Sample-effizienter, aber komplexer

</details>

---

### Frage 50
**Was ist der Unterschied zwischen Model-Based und Model-Free Imitation Learning?**

<details>
<summary>Antwort anzeigen</summary>

**Model-Based vs Model-Free IL:**

| Aspekt | Model-Based | Model-Free |
|--------|-------------|------------|
| **Modell** | Lernt Environment-Modell | Kein Modell |
| **Planning** | Kann planen | Direkte Policy-Lernen |
| **Beispiele** | GAIL (indirect), Model-Based IL | Behavioral Cloning, DAgger |

**Model-Based IL:**
- Lernt Transition Dynamics
- Kann simulieren und planen
- Oft mit IRL kombiniert

**Model-Free IL:**
- Lernt Policy direkt von Demonstrationen
- Einfacher, aber anfällig für Distributional Shift
- BC, DAgger, etc.

</details>

---

## Wichtige Formeln (auswendig lernen!)

### Transformers
```
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
```

### LSTM (6 Gleichungen)
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

### Parameter Count
```
LSTM: N = 4 × d_h × (d_x + d_h + 1)
GRU:  N = 3 × d_h × (d_x + d_h + 1)
```

### Q-Learning
```
Q(s,a) ← Q(s,a) + α[r + γ·max_{a'}Q(s',a') - Q(s,a)]
```

### Double DQN
```
a* = argmax_{a'} Q(s',a'; θ)
y = r + γ · Q(s',a*; θ^-)
```

### UCB
```
UCB(s,a) = Q̂(s,a) + c·√(ln N(s) / N(s,a))
```

### Bellman Optimality
```
Q*(s,a) = Σ_{s',r} p(s',r|s,a) · [r + γ·max_{a'}Q*(s',a')]
```

### VAE Reparametrisierung
```
z = μ(x) + σ(x) ⊙ ε,  ε ~ N(0,I)
```

### SHAP
```
φ_j = Σ_{S⊆N\{j}} [|S|!(|N|-|S|-1)! / |N|!] · [v(S∪{j}) - v(S)]
```

### DPO
```
L_DPO = -E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 45-50 | 🟢 Exzellent - Klausur-Ready! |
| 40-44 | 🟢 Sehr gut - Kleine Lücken schließen |
| 35-39 | 🟡 Gut - Wiederholung empfohlen |
| 30-34 | 🟡 Befriedigend - Schwache Bereiche wiederholen |
| 25-29 | 🟡 Ausreichend - Intensives Lernen nötig |
| <25 | 🔴 Nachholbedarf - Komplette Wiederholung |

---

## Themen-Checkliste

Markieren Sie die Themen, bei denen Sie unsicher waren:

- [ ] Transformers & Self-Attention
- [ ] Word Embeddings (Word2Vec, FastText, BPE)
- [ ] LSTM & RNNs
- [ ] Seq2Seq & Attention
- [ ] Q-Learning & DQN
- [ ] Double DQN & Target Networks
- [ ] Exploration (UCB, Thompson Sampling)
- [ ] GANs & Mode Collapse
- [ ] VAEs & Reparametrisierung
- [ ] Diffusion Models
- [ ] XAI (LIME, SHAP, PFI)
- [ ] Imitation Learning (BC, DAgger)
- [ ] RLHF & DPO

---

**Viel Erfolg bei der Klausur!** 🎯🎓

*Erstellt: 18.03.2026 | Tag 7: Gesamtwiederholung*

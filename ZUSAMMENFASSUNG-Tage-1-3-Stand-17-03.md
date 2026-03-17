# Zusammenfassung: Advanced ML Klausurvorbereitung (Tage 1-3, Stand 17.03.2026)

**Autor:** David (Operative Orchestrator)  
**Adressat:** Corvin (KI-Experte)  
**Datum:** 17.03.2026  
**Lernplan-Status:** Tag 3 abgeschlossen  

---

## Einleitung

Diese Zusammenfassung dokumentiert den Lernstoff der ersten drei Tage des 9-Tage-Lernplans zur Advanced ML Klausur (24.03.2026). Die behandelten Themen bilden das fundamentale Verständnis für moderne Deep-Learning-Architekturen: von den mathematischen Grundlagen über sequentielle Verarbeitung (RNNs/LSTMs) bis hin zur Transformer-Architektur, die den aktuellen State-of-the-Art in NLP repräsentiert.

Die drei Tage folgten einer bewusst rückwärts gerichteten Progression:
- **Tag 1 (15.03.):** Transformer & LLMs (modernste Architektur)
- **Tag 2 (16.03.):** RNNs, LSTMs & Word Embeddings (Vorläufer und Grundlagen)
- **Tag 3 (17.03.):** ML-Grundlagen & Backpropagation (fundamentale Mathematik)

Diese Struktur ermöglicht ein "Top-Down"-Verständnis: Zuerst das Ziel (Transformer), dann die evolutionären Vorstufen (RNNs), schließlich die mathematischen Fundamente (Backpropagation, Gradient Descent).

---

## Hauptteil: Zentrale Konzepte und Methoden

### 1. Mathematische Fundamente (Tag 3)

#### 1.1 Gradient Descent und Optimierung
Die zentrale Optimierungsmethode im Deep Learning:
```
θ_j ← θ_j - α · ∂J(θ)/∂θ_j
```
- **Lernrate α:** Kritischer Hyperparameter (zu groß → Divergenz, zu klein → langsame Konvergenz)
- **Kostenfunktion J(θ):** Least Squares für Regression, Cross-Entropy für Klassifikation

#### 1.2 Bias-Variance Tradeoff
Fundamentale Diagnose für Modellkomplexität:
- **High Bias (Underfitting):** J_train und J_CV beide groß → Modell zu einfach
- **High Variance (Overfitting):** J_train klein, J_CV groß → Modell zu komplex
- **Regularisierung:** λ-Parameter steuert Strafterm für große Gewichte
  ```
  J(θ) = (1/2n) Σ (h_θ(x^(i)) - y^(i))^2 + λ Σ θ_j^2
  ```

#### 1.3 Backpropagation
Algorithmus zur effizienten Gradientenberechnung in tiefen Netzen:
- **Kernidee:** Kettenregel, angewendet auf computation graph
- **Fehlerterm δ:** Wird layerweise rückwärts propagiert
  ```
  δ_j^(l) = Σ_i δ_i^(l+1) · Θ_ij^(l) · g'(z_j^(l))
  ∂J/∂Θ_ji^(l) = δ_j^(l+1) · a_i^(l)
  ```
- **Aktivolationsfunktionen:** ReLU (kein Sättigungsproblem für x>0), Sigmoid/tanh (Sättigung bei |x|>>0), Softmax (Mehrklassen)

---

### 2. Word Embeddings und Distributional Semantics (Tag 2)

#### 2.1 Distributional Hypothesis
"You shall know a word by the company it keeps" – semantische Ähnlichkeit durch kontextuelle Kookkurrenz.

#### 2.2 Word2Vec (2013)
Zwei Trainingsvarianten mit unterschiedlicher Zielrichtung:
- **CBOW (Continuous Bag-of-Words):** Vorhersage Zielwort aus Kontext
- **Skip-gram:** Vorhersage Kontextwörter aus Zielwort
- **Embedding-Matrix M ∈ ℝ^(V×d):** Jede Spalte = d-dimensionaler Wortvektor (d≈100-300)
- **Eigenschaft:** Vektorarithmetik erfasst semantische Relationen (king - man + woman ≈ queen)

#### 2.3 FastText und Subword-Embeddings
- **Word2Vec-Limitation:** Keine Behandlung von OOV-Wörtern (Out-of-Vocabulary)
- **FastText-Lösung:** Wörter als Bag of Character n-grams
- **Vorteil:** Produktive Morphologie, robust gegenüber Rechtschreibvariation

#### 2.4 Byte Pair Encoding (BPE)
Subword-Tokenisierung für kontrollierte Vokabulargröße:
1. Initialisiere Vokabular mit allen Zeichen
2. Iterativ: Füge häufigstes N-Gramm hinzu
3. Stoppe bei Ziel-Vokabulargröße
- **Anwendung:** WordPiece (Google), SentencePiece (Framework)

---

### 3. Recurrent Neural Networks (Tag 2)

#### 3.1 Vanilla RNN
Sequenzielle Verarbeitung mit geteilten Parametern:
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
y_t = g(W_y · h_t + b_y)
```
- **Weight Sharing:** Gleiche Parameter über alle Zeitschritte
- **Problem:** Sequentiell → schlechte Parallelisierbarkeit
- **Vanishing/Exploding Gradient:** Instabilität bei langen Sequenzen

#### 3.2 LSTM (Long Short-Term Memory)
Expliziter Memory-Mechanismus zur Lösung des Vanishing-Gradient-Problems:

**Drei Gates steuern Informationsfluss:**
1. **Forget Gate:** Entscheidet was aus Cell State gelöscht wird
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```
2. **Input Gate:** Entscheidet was neue Information hinzugefügt wird
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```
3. **Output Gate:** Filtert Output aus Cell State
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   ```

**Cell State Update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
h_t = o_t ⊙ tanh(C_t)
```

#### 3.3 GRU (Gated Recurrent Unit)
Vereinfachte LSTM-Alternative:
- **Kein separater Cell State** (nur Hidden State)
- **Two Gates:** Update Gate (z) + Reset Gate (r)
- **Formel:**
  ```
  z_t = σ(W_z · [h_{t-1}, x_t])
  r_t = σ(W_r · [h_{t-1}, x_t])
  h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ tanh(W · [r_t ⊙ h_{t-1}, x_t])
  ```
- **Trade-off:** Weniger Parameter, schnelleres Training, oft vergleichbare Performance

#### 3.4 Seq2Seq und Attention
**Encoder-Decoder-Architektur:**
- Encoder: Verarbeitet Input-Sequenz → Hidden State (Kontext-Vektor)
- Decoder: Generiert Output-Sequenz aus Hidden State

**Bottleneck-Problem:** Gesamter Satzinhalt in festem Vektor → Informationsverlust bei langen Sequenzen

**Bahdanau Attention (Lösung):**
- Decoder hat Zugriff auf alle Encoder-Hidden-States
- Berechnet gewichteten Context Vector:
  ```
  α_{t,i} = softmax(score(s_{t-1}, h_i))
  c_t = Σ_i α_{t,i} · h_i
  ```
- **Effekt:** Direkte Verbindung zwischen beliebigen Input-Output-Positionen

---

### 4. Transformer-Architektur (Tag 1)

#### 4.1 Motivation: Drei Desiderata
Transformers adressieren fundamentale RNN-Limitationen:
1. **Minimale Rechenkomplexität pro Layer**
2. **Minimale Pfadlänge** zwischen Wortpaaren (O(1) statt O(n))
3. **Maximale Parallelisierbarkeit** (keine sequentiellen Abhängigkeiten)

#### 4.2 Self-Attention Mechanismus
Kerninnovation: Jedes Token interagiert mit jedem anderen Token.

**Berechnung:**
```
q_i = W_Q · x_i    (Query)
k_i = W_K · x_i    (Key)
v_i = W_V · x_i    (Value)

Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Komponenten:**
- **QK^T:** Skalarprodukt bestimmt Relevanz zwischen Token-Paaren
- **√d_k:** Skalierung verhindert Softmax-Sättigung bei großen d_k
- **softmax:** Normalisierung zu Attention-Wahrscheinlichkeiten

**Komplexität:** O(n² · d) – quadratisch in Sequenzlänge, aber hoch parallelisierbar

#### 4.3 Multi-Head Attention
Parallelisierung des Attention-Mechanismus:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```
- **h Heads:** Typisch 8-16
- **d_k = d/h:** Reduzierte Dimension pro Head
- **Semantik:** Verschiedene Heads lernen unterschiedliche Relationstypen (syntaktisch, semantisch, coreferentiell)

#### 4.4 Positional Encoding
Attention ist permutationsinvariant → Reihenfolgeinformation muss explizit hinzugefügt werden.

**Sinus/Cosinus-Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- **Additiv:** x' = e + PE(pos)
- **Eigenschaft:** Unterschiedliche Frequenzen pro Dimension → eindeutige Kodierung

**Moderne Alternativen:**
- **RoPE (Rotary Positional Embedding):** Relative Position durch Rotation von Q und K
- **ALiBi:** Bias basierend auf relativer Position
- **Vorteil:** Bessere Extrapolation zu längeren Context Windows

#### 4.5 Encoder vs. Decoder Architekturen

| Aspekt | Encoder-only (BERT) | Decoder-only (GPT) |
|--------|---------------------|-------------------|
| **Attention** | Bidirektional (alle Tokens sehen alle) | Kausal (nur vergangene Tokens) |
| **Pretraining** | Masked Language Model + NSP | Next-Token Prediction |
| **Output** | Kontext-Embeddings | Token-Wahrscheinlichkeiten |
| **Use Case** | Understanding (Klassifikation, QA, NER) | Generierung (Chat, Code, Text) |

**Masked Attention im Decoder:**
- Attention-Werte für zukünftige Tokens auf -∞ gesetzt (vor Softmax)
- Ermöglicht autoregressive Generierung

#### 4.6 Feed Forward Network (FFN)
Position-wise Verarbeitung ohne Token-Interaktion:
```
FFN(x) = W_2 · GELU(x·W_1 + b_1) + b_2
```
- **Typisch:** Hidden Dimension 4× Input-Dimension
- **Modern:** SwiGLU statt ReLU/GELU (State-of-the-Art bei LLMs)
  ```
  SwiGLU(x) = W_2 · (Swish(x·W + b) ⊙ (x·V + c))
  ```

#### 4.7 Residual Connections und Layer Normalization
Stabilisierung tiefen Trainings:
- **Residual:** a_l = F(a_{l-1}) + a_{l-1} (Gradientenfluss)
- **LayerNorm:** LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
- **Pre-Norm:** Moderner Standard (LayerNorm vor Sub-Layer → stabileres Training)

---

### 5. Finetuning und Alignment (Tag 1)

#### 5.1 Parameter-Efficient Finetuning (PEFT)
**LoRA (Low-Rank Adaptation):**
- **Idee:** Gewichte eingefroren, nur low-rank Update trainiert
  ```
  W' = W + ΔW = W + B·A
  ```
  - B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
- **Vorteil:** ~0.1% der Parameter trainierbar, kein Inference-Overhead (B·A kann zu W addiert werden)

**QLoRA:** 4-bit Quantization + CPU-RAM für Optimizer States → Training großer Modelle auf einer GPU

#### 5.2 RLHF (Reinforcement Learning from Human Feedback)
Alignment-Pipeline für menschliche Präferenzen:
1. **Sammle Feedback:** Menschliche Rankings von Modelloutputs
2. **Trainiere Reward-Modell:** Vorhersagt Präferenzen
   ```
   L = -E[log σ(R(x, y_w) - R(x, y_l))]
   ```
3. **RL-Optimierung:** PPO mit KL-Strafe zur ursprünglichen Policy

**Herausforderung:** Reward Hacking (Modell exploitiert Reward-Funktion)

#### 5.3 DPO (Direct Preference Optimization)
Vereinfachte Alternative zu RLHF:
- **Kein separates Reward-Modell nötig**
- **Direkte Optimierung auf Präferenzen:**
  ```
  L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
  ```
- **Vorteil:** Gleiche/bessere Performance, erheblich einfacher (kein RL)

---

## Verbindungen zwischen den Themen

### Evolutionäre Kette: RNNs → Attention → Transformer

1. **RNN-Limitation:** Sequentielle Verarbeitung, vanishing gradient bei langen Abhängigkeiten
2. **LSTM-Lösung:** Expliziter Memory-Mechanismus (Cell State + Gates)
3. **Seq2Seq-Bottleneck:** Fester Kontext-Vektor für gesamte Sequenz
4. **Bahdanau Attention:** Decoder sieht alle Encoder-States gewichtet
5. **Self-Attention Generalisierung:** Attention innerhalb derselben Sequenz (nicht nur Encoder→Decoder)
6. **Transformer:** Vollständige Replacement von Rekurrenz durch Self-Attention

### Embedding-Evolution: One-Hot → Dense → Contextual

1. **BoW/TF-IDF:** Sparse, keine Semantik, Reihenfolge ignoriert
2. **Word2Vec:** Dense Vektoren, Distributional Hypothesis, Analogien
3. **FastText:** Subword-Information, OOV-Behandlung
4. **Transformer-Embeddings:** Contextual (gleiches Wort, unterschiedliche Bedeutung je Kontext)

### Mathematische Kontinuität: Backpropagation → BPTT → Transformer-Backprop

1. **Backprop (MLP):** Kettenregel auf computation graph
2. **BPTT (RNN):** Backprop über Zeitschritte (weight sharing)
3. **Transformer:** Backprop durch Attention-Layer (Q,K,V-Matrizen)

---

## Wichtige Formeln (Klausurauswendiglernung)

| Konzept | Formel |
|---------|--------|
| **Gradient Descent** | θ_j ← θ_j - α · ∂J(θ)/∂θ_j |
| **Regularisierung** | J(θ) = (1/2n) Σ (h_θ(x^(i)) - y^(i))^2 + λ Σ θ_j^2 |
| **Backprop Error** | δ_j^(l) = Σ_i δ_i^(l+1) · Θ_ij^(l) · g'(z_j^(l)) |
| **Self-Attention** | Attention(Q,K,V) = softmax(QK^T/√d_k) · V |
| **LSTM Forget Gate** | f_t = σ(W_f · [h_{t-1}, x_t] + b_f) |
| **LSTM Cell State** | C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t |
| **LSTM Output** | h_t = o_t ⊙ tanh(C_t) |
| **Attention Score** | α_{t,i} = softmax(score(s_{t-1}, h_i)) |
| **Context Vector** | c_t = Σ_i α_{t,i} · h_i |
| **Positional Encoding** | PE(pos,2i) = sin(pos/10000^(2i/d)) |
| **LayerNorm** | LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β |
| **LoRA Update** | W' = W + B·A, r << min(d,k) |
| **DPO Loss** | L_DPO = -E[log σ(β·log(π_θ(y_w)/π_ref) - β·log(π_θ(y_l)/π_ref))] |

---

## Typische Klausurfragen (Prüfungsrelevanz)

### Konzeptuelle Fragen ("Erklären Sie...")
1. **Self-Attention:** Beschreiben Sie Q, K, V und deren Berechnung!
2. **LSTM Gates:** Erklären Sie die Funktion der drei Gates!
3. **Encoder vs. Decoder:** Was ist der Unterschied in Attention und Training?
4. **Positional Encoding:** Warum notwendig, wie funktioniert es?
5. **Bias-Variance Tradeoff:** Diagnose und Gegenmaßnahmen!

### Vergleichsfragen ("Unterschied zwischen...")
6. **CBOW vs. Skip-gram:** Trainingsrichtung und Anwendungsfälle
7. **LSTM vs. GRU:** Architektonische Unterschiede, Trade-offs
8. **RNN vs. Transformer:** Parallelisierbarkeit, langfristige Abhängigkeiten
9. **Encoder-only vs. Decoder-only:** Use Cases, Pretraining-Ziele
10. **RLHF vs. DPO:** Komplexität, Performance, Implementierung

### Algorithmische Fragen ("Wie funktioniert...")
11. **Backpropagation:** Fehler-Rückwärtspropagation durch Netz
12. **Multi-Head Attention:** Parallelisierung, Concatenation
13. **LoRA:** Low-Rank Adaptation, Parameter-Effizienz
14. **Bahdanau Attention:** Context Vector, Alignment Scores
15. **Seq2Seq mit Attention:** Encoder-Decoder, Bottleneck-Lösung

### Begründungsfragen ("Warum...")
16. **Warum √d_k Skalierung?** Verhindert Softmax-Sättigung
17. **Warum LSTM statt RNN?** Vanishing Gradient Problem
18. **Warum Transformer statt RNN?** Parallelisierbarkeit, O(1) Pfadlänge
19. **Warum LoRA funktioniert?** Low-rank Struktur von Gewicht-Updates
20. **Warum Pre-Norm stabiler?** Gradientenfluss vor Sub-Layer

---

## Nicht klausurrelevant (laut PROJECT-File)

❌ Konkrete Tokenisierungsmethoden (BPE-Details, WordPiece-Algorithmus)  
❌ Aktivierungsfunktionen-Details (nur ReLU, GELU, SwiGLU kennen)  
❌ Positional Encoding Formeln auswendig (Zweck bekannt, Formel nicht)  
❌ Metriken & Benchmarks (GLUE, SuperGLUE, HELM)  
❌ Layer-wise Relevance Propagation, Integrated Gradients (Details)  
❌ Wasserstein GAN  
❌ RL: DDPG, TRPO, PPO, SAC (Implementierungstipps)  
❌ Information Gain in Deep RL (nur Konzept)  

---

## Offene Fragen für Dozenten-Termin (Dienstag 17.03.)

Basierend aus 01_VORBEREITUNG-Dienstag.md:

1. **Self-Attention Mathematik:** "Wie genau werden Attention-Scores berechnet? Ist das Detail oder Konzept?"
2. **LSTM Gates:** "Unterschied forget/input/output gate – klausurrelevant?"
3. **Q-Learning vs. Policy Gradients:** "Faustregeln für Auswahl?"
4. **Detailtiefe Transformer:** "Müssen wir Architektur skizzieren oder reicht Konzept?"
5. **Typische Klausurfehler:** "Was wird oft falsch verstanden?"
6. **XAI (LIME, SHAP):** "Grundidee oder mathematische Details?"

---

## Lernplan-Status (Tage 1-3)

| Tag | Datum | Thema | Status | Quelle |
|-----|-------|-------|--------|--------|
| 1 | 15.03. | Transformers & Self-Attention | ✅ Abgeschlossen | 05_ZUSAMMENFASSUNG-03 |
| 2 | 16.03. | LSTMs, RNNs, Word Embeddings | ✅ Abgeschlossen | 04_ZUSAMMENFASSUNG-02 |
| 3 | 17.03. | ML-Grundlagen, Backprop | ✅ Abgeschlossen | 03_ZUSAMMENFASSUNG-01 |

**Nächste Schritte (Tag 4-9):**
- Tag 4 (18.03.): Reinforcement Learning Teil 1 (Q-Learning, DQN)
- Tag 5 (19.03.): RL Teil 2 (Exploration, Offline RL)
- Tag 6 (20.03.): GANs, VAEs, Diffusion
- Tag 7 (21.03.): XAI + Imitation Learning
- Tag 8 (22.03.): Wiederholung + Quiz
- Tag 9 (23.03.): Letzte Vorbereitung
- Tag 10 (24.03.): Klausur

---

## Zusammenfassung der Kernkonzepte

**Fundament:** Backpropagation + Gradient Descent + Bias-Variance Tradeoff  
**Sequenzverarbeitung:** RNN → LSTM (Gates) → GRU (vereinfacht) → Attention (Bahdanau)  
**Moderne Architektur:** Transformer (Self-Attention, Multi-Head, Positional Encoding)  
**Praxis-Anwendung:** Finetuning (LoRA), Alignment (RLHF, DPO)  

**Evolutionäre Einsicht:** Jede Architektur adressiert Limitationen der Vorgänger:
- LSTM löst vanishing gradient von RNN
- Attention löst Seq2Seq-Bottleneck
- Transformer löst sequentielle Abhängigkeit von RNN
- LoRA löst Parameter-Ineffizienz von Full Finetuning
- DPO löst Komplexität von RLHF

---

*Diese Zusammenfassung dient als Klausur-Exposé und Referenzdokument für die weitere Lernplanung. Alle Formeln und Konzepte sind prüfungsrelevant gemäß 00_PROJECT-Advanced_ML.md.*

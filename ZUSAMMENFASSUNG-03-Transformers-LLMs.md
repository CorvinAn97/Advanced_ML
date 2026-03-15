# ZUSAMMENFASSUNG 03: Transformer Modelle und Large Language Models

## Übersicht
- Seitenzahl: ~80 Seiten (20MB PDF - sehr ausführlich!)
- Hauptthemen: Transformer-Architektur, Self-Attention, Training, Finetuning, PEFT, RLHF, DPO, Reasoning

## Detaillierte Inhalte

### 1. Motivation für Transformers (vs RNNs)

#### Drei Desiderata bei der Entwicklung:
1. **Minimierung der Rechenkomplexität pro Schicht**
2. **Minimierung der Pfadlänge** zwischen beliebigen Wortpaaren (für langfristige Abhängigkeiten)
3. **Maximierung parallelisierbarer Berechnungen**

#### Probleme von RNNs:
- O(Sequenzlänge) Schritte bevor entfernte Wörter interagieren
- Schwer langfristige Abhängigkeiten zu lernen (vanishing gradient)
- Sequentielle Abhängigkeit: O(n) nicht-parallelisierbare Operationen
- GPUs/TPUs können Potenzial nicht ausschöpfen

#### Transformer-Vorteile:
- O(1) maximale Interaktionsdistanz (jeder mit jedem)
- O(n²) Komplexität pro Layer, aber gut parallelisierbar
- Training auf extrem großen Datensätzen möglich

---

### 2. Self-Attention Mechanismus

#### Grundidee
- Jedes Wort interagiert mit jedem anderen Wort des Inputs
- Query, Key, Value aus Input-Embeddings berechnet:
```
q_i = W_Q · x_i
k_i = W_K · x_i
v_i = W_V · x_i
```

#### Attention Scores berechnen
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```
- **QK^T:** Skalarprodukt zwischen Query und allen Keys
- **√d_k:** Skalierung (damit Softmax nicht in Sättigung geht)
- **softmax:** Normalisiert zu Wahrscheinlichkeiten

#### Dimensionsbetrachtung
- Q, K, V: jeweils d × d matrices (d = Embedding-Dimension)
- Komplexität: O(n² · d) - quadratisch in Sequenzlänge!
- Sehr großes Context Window wird ohne Tricks teuer/langsam

---

### 3. Multi-Head Attention

#### Idee
- Mehrere Attention-Heads parallel ausführen
- Jeder Head kann sich auf verschiedene Aspekte/Beziehungen fokussieren

#### Implementierung
```
MultiHeadAtt = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```
- h = Anzahl Heads (typisch 8-16)
- d_k = d/h Dimension pro Head
- Ähnlich zu verschiedenen Filtern in CNN

---

### 4. Feed Forward Network (FFN)

#### Position-wise FFN
- Wird auf jedes Token separat angewendet
- Keine Interaktion zwischen Tokens (im Gegensatz zu Attention)
- Typischerweise 1 Hidden Layer mit 4× Input-Dimension

#### Varianten
**Klassisch (ReLU/GELU):**
```
FFN(x) = W_2 · GELU(x·W_1 + b_1) + b_2
```

**SwiGLU (State-of-the-Art bei LLMs):**
```
SwiGLU(x) = W_2 · (Swish(x·W + b) ⊙ (x·V + c))
```
- Gate-Mechanismus: Situationsabhängiges Durchlassen von Features
- Swish/SiLU: z · σ(z)
- Zwei Projektionen im Hidden Layer

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x · Φ(x)  (Φ = CDF der Standardnormalverteilung)
```
- Approximation: x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
- "Softer" Cutoff als ReLU, besserer Gradientenfluss

---

### 5. Positional Encoding

#### Problem
- Attention hat keine Information über Wortreihenfolge
- "Mann isst Dinosaurier" ≠ "Dinosaurier isst Mann"

#### Lösung: Sinus/Cosinus Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- pos: Position im Satz
- i: Dimension des Encodings
- d: Modell-Dimension
- Wird zu Word Embedding addiert: x' = e + PE(pos)

#### Eigenschaften
- Eindeutig über alle Dimensionen (verschiedene Frequenzen)
- Periodisch, aber Periodenlänge sehr groß
- Nicht lernbar (im ursprünglichen Transformer)

#### Alternativen
- **RoPE (Rotary Positional Embedding):** Relative Position durch Rotation von Q und K
- **ALiBi:** Bias basierend auf relativer Position
- Bessere Extrapolation zu längeren Sequenzen

---

### 6. Transformer Encoder-Decoder Architektur

#### Encoder
- Self-Attention (alle Tokens sehen alle anderen)
- Feed Forward Network
- N Layer wiederholt
- Output: Kontextbezogene Repräsentationen

#### Decoder
- **Masked Self-Attention:** Nur vergangene Tokens sichtbar (kausal)
- **Cross-Attention:** Q aus Decoder, K,V aus Encoder
- Feed Forward Network
- Linear + Softmax für Token-Vorhersage

#### Masking im Decoder
- Attention-Werte für zukünftige Tokens auf -∞ setzen (vor Softmax)
- Sorgt für autoregressive Generierung
- Technisch: Masked Multi-Head Attention

---

### 7. Residual Connections und Layer Normalization

#### Residual Connections
```
a_l = F(a_{l-1}) + a_{l-1}
```
- Hilft bei Informations- und Gradientenfluss
- Macht tiefe Netze trainierbar
- "Shortcut" um Layer herum

#### Layer Normalization
```
LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```
- Normalisiert Aktivierungen pro Datenpunkt über alle Features
- Unabhängig von Batchgröße
- γ, β lernbare Parameter (Skalierung und Verschiebung)

#### Pre-Norm vs Post-Norm
- **Post-Norm (Original):** LayerNorm nach Sub-Layer
- **Pre-Norm (Modern):** LayerNorm vor Sub-Layer - stabileres Training

---

### 8. Tokenisierung und Embeddings

#### Subword-Tokenisierung
- **Ziel:** Offener Wortschatz bei begrenzter Vokabulargröße
- **Byte Pair Encoding (BPE):** Häufige Zeichenfolgen iterativ zusammenführen
- **WordPiece:** Wählt Subwords nach Modellwahrscheinlichkeit
- **SentencePiece:** Framework, arbeitet auf Rohtext
- **Typische Vokabulargröße:** 30k-100k Tokens

#### Tokenizer Beispiel
```
Text → Tokens → Token IDs → Embeddings
"Hello world" → ["Hello", " world"] → [15496, 995] → E[15496], E[995]
```

#### Embedding Layer
```
E ∈ ℝ^(V×d)  (V = Vokabulargröße, d = Embedding-Dimension)
```
- Jede Zeile = Embedding eines Tokens
- Vokabulargröße bestimmt direkt Modellgröße

---

### 9. Encoder-only vs Decoder-only Modelle

#### Encoder-only (z.B. BERT)
- Bidirektionale Attention
- Pretraining: Masked Language Model + Next Sentence Prediction
- **Zweck:** Sprachrepräsentationen für Understanding-Tasks
- **Typische Tasks:** Klassifikation, QA, NER, Similarity

#### Decoder-only (z.B. GPT)
- Kausale (Masked) Attention
- Pretraining: Next-Token Prediction
- **Zweck:** Textgenerierung
- **Typische Tasks:** Chatbots, Code-Generierung, Dialog

#### Vergleich
| Aspekt | Encoder-only | Decoder-only |
|--------|--------------|--------------|
| Attention | Bidirektional | Kausal |
| Training | Masked LM | Next-Token |
| Output | Embeddings | Token-Wahrscheinlichkeiten |

---

### 10. Transformer Parameterzählung

#### Beispiel BERT-Base
- Vokabular: 30,522
- Embedding-Dimension: d = 768
- Heads: h = 12
- Layer: L = 12
- FFN Hidden: d_ff = 3072

**Parameter pro Layer:**
- Self-Attention: 4d² + 4d ≈ 2.36M
- FFN: (d+1)·d_ff + (d_ff+1)·d ≈ 4.72M
- Gesamt pro Layer: ~7.08M
- Alle Layer: L × 7.08M = 85M
- Embeddings: 30,522 × 768 = 23.4M
- **Gesamt: ~110M Parameter**

---

### 11. Finetuning von LLMs

#### In-Context Learning
- **Zero-shot:** Aufgabe im Prompt ohne Beispiele
- **One/Few-shot:** Beispiele im Prompt
- Limitiert durch Context Window

#### Instruction Finetuning
- Paare aus Prompt (Instruction) und Completion
- Gewichte des Modells werden angepasst
- **Catastrophic Forgetting:** Performance auf anderen Tasks kann sinken

#### Full Finetuning vs PEFT
- **Full Finetuning:** Alle Parameter trainieren (teuer)
- **PEFT:** Nur wenige Parameter trainieren (effizient)

---

### 12. Parameter-Efficient Finetuning (PEFT)

#### LoRA (Low-Rank Adaptation)
**Idee:**
```
W' = W + ΔW = W + B·A
```
- W wird eingefroren
- ΔW = B·A mit Rang r << min(d,k)
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
- Trainiere nur B und A

**Vorteile:**
- ~0.1% der Parameter trainierbar
- Gleiche Performance wie Full Finetuning
- Kein Inference-Overhead (B·A kann zu W addiert werden)

#### QLoRA
- 4-bit Quantization statt 16-bit
- Gradienten und Optimizer States in CPU-RAM
- Ermöglicht Training großer Modelle auf einer GPU

#### Prompt Tuning
- Füge trainierbare "Soft Prompts" (s_1, ..., s_n) zu Input hinzu
- Modellgewichte eingefroren
- Nur Soft Prompts trainieren (10k-100k Parameter)
- Erreicht bei großen Modellen (10B+) Performance von Full Finetuning

---

### 13. Reinforcement Learning from Human Feedback (RLHF)

#### Pipeline
1. **Sammle menschliches Feedback:** Rankings von Modelloutputs
2. **Trainiere Reward-Modell:** Vorhersagt menschliche Präferenzen
3. **Optimiere LLM mit RL:** Maximiere Reward (z.B. mit PPO)

#### Reward Modell Training
- Paarweise Vergleiche: (Prompt x, bevorzugte Antwort y_w, schlechtere y_l)
- Loss: -log σ(R(x, y_w) - R(x, y_l))

#### RL Training des LLM
- Policy π_θ = LLM
- Reward r = Reward-Modell(x, y)
- Zusätzlich: KL-Divergenz zur ursprünglichen Policy (verhindert Reward Hacking)
- Algorithmus: PPO (Proximal Policy Optimization)

#### Herausforderungen
- **Reward Hacking:** Modell findet Schlupflöcher im Reward
- **Toxizität:** Kann unerwünschtes Verhalten erzeugen

---

### 14. Direct Preference Optimization (DPO)

#### Idee
- Kein separates Reward-Modell nötig
- Optimiere LLM direkt auf Präferenzen

#### Formel
```
L_DPO(π_θ; π_ref) = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```
- π_θ: Aktuelles Modell
- π_ref: Referenz-Modell (SFT-Modell)
- β: Temperatur-Parameter

#### Vorteile
- Einfacher als RLHF (kein RL nötig)
- Gleiche oder bessere Performance

---

### 15. Constitutional AI

#### Idee
- LLM beaufsichtigt sich selbst anhand von Regeln (Constitution)
- Keine menschliche Intervention nötig

#### Schritte
1. **SL-CAI:** Self-Critique und Revision → Supervised Finetuning
2. **RL-CAI:** RL from AI Feedback (RLAIF)
- PM (Preference Model) trainiert auf AI-Rankings
- RL-Training mit diesem PM

---

### 16. Reasoning in LLMs

#### Chain-of-Thought Prompting
- LLM Schritt-für-Schritt denken lassen
- Beispiele zeigen den gewünschten Reasoning-Stil
- Bessere Performance bei komplexen Problemen

#### Advanced Reasoning
- **Tree of Thoughts:** Mehrere Gedankenpfade erkunden
- **Self-Consistency:** Mehrere Samples generieren, Mehrheitsentscheid
- **Tool Use:** LLM nutzt externe Tools (Rechner, Suche)

---

### 17. Interaktion mit externen Datenbanken

#### Retrieval-Augmented Generation (RAG)
1. Nutzer-Query → Embedding
2. Ähnlichkeitssuche in Vector Database
3. Gefundene Dokumente + Query → LLM
4. LLM generiert Antwort basierend auf Kontext

#### Vorteile
- Aktuelle Informationen (kein Training nötig)
- Quellenangaben möglich
- Reduziert Halluzinationen

#### Tool Use / Function Calling
- LLM kann externe Tools aufrufen
- Beispiele: API-Aufrufe, Datenbankabfragen, Code-Ausführung
- Format: JSON mit Tool-Namen und Parametern

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Self-Attention Mechanismus
- Warum: Kern des Transformers
- Was: Q, K, V Berechnung, Attention Scores, Softmax, Skalierung

### ✅ Multi-Head Attention
- Warum: Ermöglicht vielfältige Beziehungen
- Was: Parallelisierung, Concatenation, verschiedene Heads

### ✅ Positional Encoding
- Warum: Reihenfolgeinformation nötig
- Was: Sinus/Cosinus Formeln, RoPE Alternative

### ✅ Encoder vs Decoder Unterschiede
- Warum: Grundlegendes Verständnis
- Was: Bidirektional vs Kausal, Masking, Cross-Attention

### ✅ Finetuning und PEFT (LoRA)
- Warum: Wichtige Praxis-Anwendung
- Was: Low-Rank Adaptation, Parameter-Effizienz

### ✅ RLHF und DPO
- Warum: Moderne Alignment-Methoden
- Was: Reward-Modell, RL-Training, DPO-Simplifikation

## Formeln/Algorithmen (wichtig)

### Self-Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
```

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Layer Normalization
```
LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

### GELU
```
GELU(x) = x · Φ(x) ≈ x · 0.5 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```

### LoRA Update
```
W' = W + B·A,  wobei B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
```

### Reward Modell Loss
```
L = -E[log σ(R(x, y_w) - R(x, y_l))]
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **Transformer:** Self-Attention ersetzt Rekurrenz
- **O(n²) Komplexität:** Skaliert schlecht mit sehr langen Sequenzen
- **Q, K, V Konzept:** Aus der Information Retrieval-Welt übernommen
- **Positional Encoding:** Notwendig da Attention position-unabhängig
- **Decoder-only:** Aktueller Standard für LLMs (GPT, Llama, etc.)
- **LoRA:** Elegante Lösung für Parameter-effizientes Finetuning
- **RLHF:** Alignment mit menschlichen Präferenzen
- **DPO:** Vereinfacht RLHF erheblich

### ⚠️ Häufige Fehler
- Attention ohne Skalierung (1/√d_k vergessen)
- Positional Encoding multiplizieren statt addieren
- Masking im Encoder (falsch - nur im Decoder)
- LoCA statt LoRA schreiben 😄

### 📝 Prüfungsrelevante Fragen
1. Wie funktioniert Self-Attention? Erklären Sie Q, K, V!
2. Warum braucht man Positional Encoding?
3. Was ist der Unterschied zwischen Encoder und Decoder?
4. Wie funktioniert Masked Attention?
5. Was ist LoRA und warum funktioniert es?
6. Erklären Sie RLHF!
7. Was ist der Vorteil von DPO gegenüber RLHF?
8. Was ist der Unterschied zwischen GELU und ReLU?
9. Warum ist Transformer-Attention O(n²)?
10. Was ist Multi-Head Attention?

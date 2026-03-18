# Selbsttest Tag 1: Transformers & Self-Attention

**Umfang:** Ausführlicher Test zu allen Themen des Tages  
**Zeitansatz:** 45-60 Minuten  
**Hinweis:** Antworten nicht einfach ablesen - erst selbst überlegen, dann nachschlagen!

---

## Teil A: Grundkonzepte & Verständnis (15 Fragen)

### 1. Motivation & Kontext

**Frage 1:**  
Erklären Sie das fundamentale Problem von RNNs/LSTMs, das durch Self-Attention gelöst wird. Warum sind parallele Berechnungen bei RNNs schwierig?

**Frage 2:**  
Was bedeutet "permutation invariant" im Kontext von Self-Attention? Warum ist das ein Problem und wie wird es gelöst?

**Frage 3:**  
Vergleichen Sie die maximale Pfadlänge (Anzahl Schritte zwischen zwei Tokens) bei RNNs vs. Transformern. Was sind die Konsequenzen?

---

### 2. Self-Attention Mechanismus

**Frage 4:**  
Erklären Sie die Rollen von Query, Key und Value in Analogie zu einer Datenbanksuche oder einem Bibliothekssystem.

**Frage 5:**  
Warum wird in der Attention-Formel durch die Wurzel aus d_k (sqrt(d_k)) dividiert? Was würde passieren, wenn man das nicht täte?

**Frage 6:**  
Beschreiben Sie Schritt für Schritt, wie der Attention-Score für Position i berechnet wird:
1. Welche Vektoren werden benötigt?
2. Welche Operationen werden durchgeführt?
3. Was ist das Endergebnis?

**Frage 7:**  
Was ist der Unterschied zwischen Self-Attention und Cross-Attention? Wann wird welche verwendet?

---

### 3. Multi-Head Attention

**Frage 8:**  
Was ist die Idee hinter Multi-Head Attention? Warum nicht einfach einen einzigen Attention-Head mit mehr Dimensionen verwenden?

**Frage 9:**  
Berechnen Sie: Bei einem Transformer mit d_model = 512 und 8 Heads - wie groß ist die Dimension d_k pro Head? Wie viele separate Attention-Berechnungen laufen parallel?

**Frage 10:**  
Welche verschiedenen Arten von Beziehungen könnten verschiedene Heads lernen? Nennen Sie 3 konkrete Beispiele bei der Verarbeitung von natürlicher Sprache.

---

### 4. Architektur: Encoder & Decoder

**Frage 11:**  
Zeichnen Sie grob den Aufbau eines Transformer-Encoders (einzelne Layer). Welche Sub-Layer gibt es und in welcher Reihenfolge?

**Frage 12:**  
Zeichnen Sie grob den Aufbau eines Transformer-Decoders. Welche zusätzlichen Komponenten gibt es im Vergleich zum Encoder?

**Frage 13:**  
Was ist der entscheidende Unterschied zwischen Encoder-Only (z.B. BERT) und Decoder-Only (z.B. GPT) Modellen?
- Wie unterscheiden sich die Attention-Mechanismen?
- Welche Art von Aufgaben ist jeweils besser geeignet?

---

### 5. Masking & Positional Encoding

**Frage 14:**  
Was ist "Masked Attention" (Causal/Autoregressive Masking) und warum wird sie im Decoder benötigt? Wie sieht die Maskierungsmatrix aus?

**Frage 15:**  
Warum benötigen Transformer Positional Encoding? Erklären Sie das Problem, das ohne Positional Encoding entstünde.

---

## Teil B: Formeln & Berechnungen (8 Fragen)

### 6. Mathematische Formulierungen

**Frage 16:**  
Schreiben Sie die Self-Attention Formel aus dem Gedächtnis auf:

```
Attention(Q, K, V) = ?
```

**Frage 17:**  
Gegeben: Q, K, V ∈ R^(n × d_k) mit n = 4 (Sequenzlänge) und d_k = 64.
- Welche Dimension hat QK^T?
- Welche dimension hat softmax(QK^T / sqrt(d_k))?
- Welche Dimension hat der finale Attention-Output?

**Frage 18:**  
Erklären Sie die Berechnung von Q, K, V aus dem Input X:
- Welche Gewichtsmatrizen werden verwendet?
- Sind W_Q, W_K, W_V unterschiedlich für jeden Layer?
- Sind sie unterschiedlich für jeden Head?

**Frage 19:**  
Was bedeutet "Scaled" Dot-Product Attention? Berechnen Sie für d_k = 64 den Skalierungsfaktor.

---

### 7. Multi-Head Berechnung

**Frage 20:**  
Beschreiben Sie die drei Phasen von Multi-Head Attention:
1. Lineare Projektionen (welche Matrizen, welche Dimensionen?)
2. Parallele Attention-Berechnung (was passiert hier?)
3. Concatenation & finale Projektion (was wird wie zusammengefügt?)

**Frage 21:**  
Warum wird nach dem Concatenieren der Heads eine finale Lineare Transformation W_O benötigt?

---

## Teil C: Vergleiche & Analyse (10 Fragen)

### 8. Transformer vs. RNN/LSTM

**Frage 22:**  
Vergleichen Sie folgende Aspekte:

| Aspekt | RNN/LSTM | Transformer |
|--------|----------|-------------|
| Rechenkomplexität pro Schicht | ? | ? |
| Parallele Berechnung | ? | ? |
| Maximale Pfadlänge | ? | ? |
| Langfristige Abhängigkeiten | ? | ? |
| Sequenzlänge | ? | ? |

**Frage 23:**  
Warum können Transformer langfristige Abhängigkeiten besser modellieren als RNNs? Erklären Sie anhand der Pfadlängen.

**Frage 24:**  
Was ist der Nachteil von Self-Attention bezüglich der Speicherkomplexität? Warum ist das ein Problem bei langen Sequenzen?

---

### 9. Encoder-Only vs Decoder-Only

**Frage 25:**  
Ordhen Sie zu: BERT, GPT, T5, RoBERTa, LLaMA, DistilBERT
- Encoder-Only: ?
- Decoder-Only: ?
- Encoder-Decoder: ?

**Frage 26:**  
Warum eignet sich BERT (Encoder-Only) gut für Classification-Aufgaben, aber nicht für Textgenerierung?

**Frage 27:**  
Warum eignet sich GPT (Decoder-Only) gut für Textgenerierung, aber schlechter für Sequence-Pair-Classification (z.B. NLI)?

**Frage 28:**  
Erklären Sie, warum Decoder-Only Modelle trotz fehlendem "Blick nach rechts" heute die dominierende Architektur für große Sprachmodelle sind.

---

## Teil D: Praktische Anwendungen & Edge Cases (7 Fragen)

### 10. Training & Inference

**Frage 29:**  
Warum wird beim Training von Decoder-Only Modellen Teacher Forcing verwendet? Was wäre das Alternative?

**Frage 30:**  
Was ist die "Exposure Bias" beim Training von autoregressiven Modellen?

**Frage 31:**  
Erklären Sie das Konzept "KV Caching" bei der Inferenz von Decoder-Only Modellen. Warum ist das wichtig für die Effizienz?

---

### 11. Grenzfälle & tiefes Verständnis

**Frage 32:**  
Was passiert, wenn der Attention-Score zwischen zwei Positionen sehr groß wird (z.B. 100) und man Softmax anwendet, ohne zu skalieren? Wie nennt man dieses Problem?

**Frage 33:**  
Warum funktioniert Self-Attention nicht mit One-Hot-Vektoren als Input? Was brauchen wir stattdessen?

**Frage 34:**  
Wie unterscheidet sich "Cross-Attention" im Decoder von "Self-Attention" im Encoder? Welche Queries, Keys und Values werden verwendet?

**Frage 35:**  
Erklären Sie den Unterschied zwischen "Masked Self-Attention" im Decoder und "Self-Attention" im Encoder.

---

## Teil E: Erweiterte Konzepte (5 Fragen)

### 12. Moderne Entwicklungen

**Frage 36:**  
Was ist der Unterschied zwischen absolutem Positional Encoding (Original Transformer) und Relative Position Representations (später entwickelt)?

**Frage 37:**  
Was ist RoPE (Rotary Position Embedding)? Warum wird es in modernen Modellen wie LLaMA verwendet?

**Frage 38:**  
Was ist die "Attention is all you need"-These? Ist sie heute noch gültig oder gibt es Gegenstimmen?

**Frage 39:**  
Erklären Sie das Konzept "Flash Attention". Welches Problem löst es?

**Frage 40:**  
Was ist "Mixture of Experts" (MoE) und wie unterscheidet es sich von Standard-Transformer-Modellen?

---

## Antworten & Lösungen

<details>
<summary>Klicken Sie hier, um die Antworten anzuzeigen</summary>

### Teil A Antworten

**A1:** RNNs haben O(n) sequentielle Schritte - Tokens können nicht parallel verarbeitet werden. Zusätzlich ist der Pfad zwischen entfernten Tokens lang (O(n)), was zu Vanishing Gradients führt. Self-Attention hat O(1) Pfadlänge zwischen allen Token-Paaren und ist parallelisierbar.

**A2:** Self-Attention ohne Positional Encoding würde dasselbe Ergebnis liefern, egal wie die Tokens permutiert werden (Reihenfolge egal). Das wird durch Positional Encoding gelöst, das Positionsinformation in die Embeddings einbettet.

**A3:** RNN: O(n) (jeder Schritt nur Nachbar erreicht). Transformer: O(1) (direkte Verbindung durch Attention). Konsequenz: Bei langen Sequenzen können RNNs langfristige Abhängigkeiten schlechter lernen.

**A4:** Query = Suchanfrage ("Was suche ich?"), Key = Index/Label ("Wie heißt das Dokument?"), Value = Inhalt ("Was steht im Dokument?"). Attention berechnet Ähnlichkeit Query-Key und gewichtet die Values danach.

**A5:** Die Skalarprodukte QK^T haben Varianz proportional zu d_k. Ohne Skalierung würden die Werte zu groß, Softmax würde in Sättigung gehen (sehr scharfe Verteilung, nahezu one-hot), Gradienten würden verschwinden.

**A6:** 
1. Q_i (Query für Position i), K (alle Keys), V (alle Values)
2. a) Skalarprodukte s_ij = Q_i · K_j^T für alle j
   b) Skalierung: s_ij / sqrt(d_k)
   c) Softmax über j: alpha_ij = softmax(s_i.)
   d) Gewichtete Summe: output_i = sum_j(alpha_ij · V_j)

**A7:** Self-Attention: Q, K, V kommen aus derselben Quelle (z.B. Encoder-Input oder Decoder-Input). Cross-Attention: Q kommt aus Decoder, K,V aus Encoder (oder externe Quelle). Wird beim Seq2Seq-Übersetzen verwendet.

**A8:** Verschiedene Heads können verschiedene Arten von Beziehungen/Syntaktische Muster lernen (Subjekt-Verb, syntaktische Rollen, referenzielle Beziehungen). Ein einzelner Head wäre überfordert, alles gleichzeitig zu erfassen.

**A9:** d_k = 512 / 8 = 64. Es laufen 8 parallele Attention-Berechnungen.

**A10:** 1) Syntaktische Beziehungen (Subjekt-Verb), 2) Koreferenz (Pronomen → Nomen), 3) Semantische Ähnlichkeit (Synonyme), 4) Positionale Beziehungen.

**A11:** Encoder: Input → [Self-Attention → Add&Norm → FeedForward → Add&Norm] × N → Output. Zwei Sub-Layer pro Block mit Residual Connections und LayerNorm.

**A12:** Decoder hat drei Sub-Layer: Masked Self-Attention, Cross-Attention (mit Encoder), FeedForward. Zusätzlich Maskierung für Autoregressivität.

**A13:** Encoder-Only: Bidirektionale Attention (sieht links und rechts). Gut für Verstehen/Classification. Decoder-Only: Causal Attention (sieht nur links). Gut für Generierung.

**A14:** Masked Attention verhindert, dass Position i auf Positionen j > i schaut ("Blick in die Zukunft"). Maske ist untere Dreiecksmatrix (1en unterhalb, 0en oberhalb der Diagonalen).

**A15:** Ohne Positional Encoding wäre das Modell permutation-invariant (Reihenfolge egal). "Hund beißt Mann" = "Mann beißt Hund". Positional Encoding fügt Positionsinformation hinzu.

### Teil B Antworten

**A16:** Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V

**A17:** QK^T: (4 × 4). Softmax-Output: (4 × 4). Finaler Output: (4 × 64).

**A18:** Q = X · W_Q, K = X · W_K, V = X · W_V. W_Q, W_K, W_V sind pro Layer verschieden, aber von allen Heads im Layer geteilt (durch verschiedene Projektionen).

**A19:** Skalierungsfaktor = 1 / sqrt(64) = 1/8 = 0.125

**A20:** 
1. Q_i = X · W_Q^i, K_i = X · W_K^i, V_i = X · W_V^i für jeden Head i
2. Jeder Head berechnet eigene Attention: head_i = Attention(Q_i, K_i, V_i)
3. Concat(head_1, ..., head_h) · W_O

**A21:** Um die Dimension wieder auf d_model zu bringen und die Information aller Heads zu kombinieren.

### Teil C Antworten

**A22:**
- Komplexität: RNN O(n), Transformer O(n²)
- Parallel: RNN nein, Transformer ja
- Pfad: RNN O(n), Transformer O(1)
- Langfristig: RNN schwierig (Vanishing), Transformer gut
- Sequenzlänge: RNN flexibel, Transformer quadratische Komplexität

**A23:** Transformer: Direkte Verbindung O(1). RNN: Information muss durch alle Zwischenschritte fließen O(n) → Gradienten verschwinden bei langen Sequenzen.

**A24:** O(n²) Speicherkomplexität für Attention-Matrix. Bei n=1000 → 1M Einträge. Problem für sehr lange Sequenzen (deshalb Windowed Attention, Sparse Attention, etc.)

**A25:** Encoder: BERT, RoBERTa, DistilBERT. Decoder: GPT, LLaMA. Encoder-Decoder: T5.

**A26:** BERT sieht bidirektionalen Kontext (gut für Verstehen), aber hat keine autoregressive Struktur für Generierung.

**A27:** GPT sieht nur linksseitigen Kontext, daher gut für links-nach-rechts Generierung, aber schlecht für Pair-Tasks, die beide Seiten brauchen.

**A28:** Durch Skalierung (viele Parameter, viele Daten) und In-Context Learning kompensieren Decoder-Only Modelle den bidirektionalen Nachteil. Außerdem sind sie effizienter zu trainieren.

### Teil D Antworten

**A29:** Teacher Forcing: Beim Training wird das echte nächste Token als Input gegeben, nicht die eigene Vorhersage. Alternative: Eigene Vorhersagen verwenden (exposure bias Problem).

**A30:** Das Modell wird nur mit echten Daten trainiert, aber bei Inferenz mit eigenen Fehlern konfrontiert. Distributional Shift zwischen Trainings- und Testverteilung.

**A31:** KV Caching: Berechnete Keys und Values werden zwischengespeichert, nicht bei jedem Schritt neu berechnet. Reduziert Komplexität bei autoregressiver Generierung.

**A32:** Softmax würde extrem scharf werden (eine Position bekommt ~1, alle anderen ~0). Problem: "Softmax Saturation" oder "Sharpness", Gradienten verschwinden.

**A33:** One-Hot hat keine semantische Struktur. Wir brauchen dichte Embeddings (z.B. Word2Vec, gelernte Embeddings), die semantische Ähnlichkeit kodieren.

**A34:** Cross-Attention: Queries aus Decoder-Input, Keys und Values aus Encoder-Output. Ermöglicht Decoder, auf Encoder-Information zuzugreifen.

**A35:** Encoder: Bidirektional (keine Maske). Decoder: Causal/Masked (nur vorherige Positionen sichtbar).

### Teil E Antworten

**A36:** Absolut: Jede Position hat feste encodierte Werte (Sinus/Cosinus). Relativ: Attention berücksichtigt relative Distanz zwischen Positionen.

**A37:** RoPE rotiert Query und Key Vektoren basierend auf ihrer relativen Position. Effizienter, bessere Extrapolation auf längere Sequenzen als absolute Encodings.

**A38:** These: Attention-Mechanismus allein reicht, keine RNNs/CNNs nötig. Heute: Weitgehend akzeptiert, aber Mischformen (z.B. Mamba, Linear Attention) werden erforscht.

**A39:** Flash Attention optimiert Speicherzugriffsmuster für Attention-Berechnung auf GPUs. Reduziert Speicherbewegung zwischen HBM und SRAM → deutlich schneller.

**A40:** MoE verwendet mehrere Feed-Forward "Experten", aber nur einen Teil davon wird pro Token aktiviert. Skaliert Modellgröße, nicht Rechenkosten.
</details>

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 36-40 | 🟢 Exzellent - Bereit für Tag 2 |
| 30-35 | 🟢 Gut - Kleine Wiederholung empfohlen |
| 24-29 | 🟡 Befriedigend - Themen wiederholen |
| 18-23 | 🟡 Ausreichend - Tag 1 wiederholen |
| <18 | 🔴 Nachholbedarf - Zusammenfassung nochmal lesen |

---

**Tipps:**
- Markieren Sie Fragen, bei denen Sie unsicher waren
- Schauen Sie sich die zugehörigen Zusammenfassungs-Abschnitte nochmal an
- Erklären Sie schwierige Konzepte einem imaginären Lernpartner (Feynman-Technik)

**Viel Erfolg!** 🎯

# Selbsttest Tag 2: LSTM, RNNs, Word Embeddings & Seq2Seq

**Umfang:** Ausführlicher Test zu allen Themen des Tages  
**Zeitansatz:** 50-70 Minuten  
**Hinweis:** Antworten nicht einfach ablesen - erst selbst überlegen, dann nachschlagen!

---

## Teil A: Textrepräsentation & Word Embeddings (12 Fragen)

### 1. Bag of Words & TF-IDF

**Frage 1:**  
Erklären Sie das fundamentale Problem von Bag-of-Words: Warum ist "Dog bites man" = "Man bites dog" in BoW? Welche Information geht verloren?

**Frage 2:**  
Berechnen Sie TF-IDF für das Wort "cat" in Dokument D1:
- D1: "The cat sat on the mat" (6 Wörter)
- D2: "The dog sat on the log" (6 Wörter)
- D3: "The cat and the dog are friends" (8 Wörter)

**Frage 3:**  
Warum ist IDF("the") = 0 in einem Korpus, wo "the" in jedem Dokument vorkommt? Was bedeutet das praktisch?

**Frage 4:**  
Vergleichen Sie BoW/TF-IDF vs. Word2Vec hinsichtlich:
- Dimension
- Sparsity
- Speicherbedarf (bei |V|=100.000, d=300)
- Semantische Ähnlichkeit

---

### 2. Word2Vec & Distributional Hypothesis

**Frage 5:**  
Erklären Sie das Distributional Hypothesis mit eigenen Worten. Warum funktioniert es?

**Frage 6:**  
Was ist der Unterschied zwischen CBOW und Skip-gram? Nennen Sie jeweils:
- Input
- Output
- Richtung der Vorhersage
- Vorteil/Nachteil

**Frage 7:**  
Erklären Sie die Analogie-Aufgabe: king - man + woman ≈ queen. Warum funktioniert das mathematisch?

**Frage 8:**  
Was ist Negative Sampling und warum wird es verwendet? Wie viele negative Beispiele werden typischerweise verwendet?

---

### 3. FastText & BPE

**Frage 9:**  
Wie behandelt FastText Out-of-Vocabulary (OOV) Wörter? Geben Sie ein konkretes Beispiel.

**Frage 10:**  
Beschreiben Sie den BPE-Algorithmus Schritt für Schritt. Was ist das Ziel?

**Frage 11:**  
Warum ist BPE ein Kompromiss zwischen Word-Level und Character-Level Tokenisierung? Was sind die Vorteile gegenüber beiden Extremen?

**Frage 12:**  
Ein unbekanntes Wort "ChatGPT" soll mit FastText verarbeitet werden (n=3). Welche n-grams werden gebildet?

---

## Teil B: RNNs & LSTM (14 Fragen)

### 4. Vanilla RNN

**Frage 13:**  
Schreiben Sie die Update-Gleichung eines Vanilla RNN aus dem Gedächtnis auf. Was bedeuten die Variablen?

**Frage 14:**  
Warum können RNNs nicht über die Zeitdimension parallelisiert werden? Erklären Sie die Datenabhängigkeitskette.

**Frage 15:**  
Berechnen Sie die Parameterzahl eines Vanilla RNN mit d_x=100 und d_h=256.

---

### 5. LSTM - Long Short-Term Memory

**Frage 16:**  
Zeichnen Sie die 3 Gates eines LSTM und beschreiben Sie ihre Funktion:
- Forget Gate: Was macht es? Wertebereich?
- Input Gate: Was macht es? Welche zwei Komponenten?
- Output Gate: Was macht es?

**Frage 17:**  
Schreiben Sie alle 6 LSTM-Gleichungen aus dem Gedächtnis auf (inklusive Cell State Update).

**Frage 18:**  
Warum löst LSTM das Vanishing Gradient Problem? Erklären Sie den "Constant Error Carousel".

**Frage 19:**  
Berechnen Sie die Parameterzahl eines LSTM mit d_x=64 und d_h=128. Zeigen Sie Ihre Rechnung.

**Frage 20:**  
Was ist der Unterschied zwischen Cell State (C_t) und Hidden State (h_t)? Welcher speichert Langzeit- und welcher Kurzzeitinformation?

**Frage 21:**  
Warum verwendet LSTM Sigmoid für Gates und tanh für Cell State Kandidaten? Warum nicht umgekehrt?

---

### 6. GRU & Bidirektionale RNNs

**Frage 22:**  
Was ist der Hauptunterschied zwischen GRU und LSTM? Wie viele Gates hat GRU?

**Frage 23:**  
Schreiben Sie die 4 GRU-Gleichungen auf (Update Gate, Reset Gate, Candidate, Hidden State).

**Frage 24:**  
Wie funktioniert ein bidirektionales RNN? Was sind die Vor- und Nachteile?

**Frage 25:**  
Warum eignet sich ein bidirektionales RNN gut für Named Entity Recognition (NER), aber nicht für autoregressive Textgenerierung?

**Frage 26:**  
Vergleichen Sie LSTM und GRU hinsichtlich Parameterzahl, Trainingsspeed und typischer Performance.

---

## Teil C: Seq2Seq & Attention (10 Fragen)

### 7. Encoder-Decoder Architektur

**Frage 27:**  
Was ist das Bottleneck-Problem in Seq2Seq ohne Attention? Warum ist das ein Problem?

**Frage 28:**  
Beschreiben Sie den Unterschied zwischen Greedy Search und Beam Search. Was ist der Vorteil von Beam Search?

**Frage 29:**  
Beam Search mit k=2: Gegeben die folgenden Wahrscheinlichkeiten nach dem ersten Wort:
- "The": 0.4
- "A": 0.3
- "My": 0.2
- "In": 0.1

Welche zwei Beams werden behalten? Berechnen Sie für den zweiten Schritt, wenn:
- P("cat"|"The") = 0.5, P("dog"|"The") = 0.3
- P("cat"|"A") = 0.6, P("dog"|"A") = 0.2

Welche Beams werden nach Schritt 2 behalten?

---

### 8. Bahdanau Attention

**Frage 30:**  
Erklären Sie Bahdanau Attention (Additive Attention). Was sind die drei Schritte?

**Frage 31:**  
Ist der Context Vector c_t in Bahdanau-Attention für alle Decoder-Schritte gleich? Begründen Sie.

**Frage 32:**  
Wie berechnet sich der Context Vector c_t? Schreiben Sie die Formel auf.

**Frage 33:**  
Was ist der Alignment Score in Bahdanau Attention? Welche Variablen werden verwendet?

**Frage 34:**  
Warum löst Attention das Bottleneck-Problem? Vergleichen Sie mit dem klassischen Seq2Seq.

**Frage 35:**  
Visualisieren Sie eine Attention-Matrix für die Übersetzung "Der Hund" → "The dog". Wie sehen die α-Werte aus?

---

## Teil D: Vergleiche & Analyse (4 Fragen)

### 9. Architektur-Vergleiche

**Frage 36:**  
Vergleichen Sie RNN, LSTM und GRU in einer Tabelle:
- Anzahl Gates
- Parameterzahl (bei gleicher Hidden-Dimension)
- Fähigkeit für lange Abhängigkeiten
- Trainingsspeed

**Frage 37:**  
Warum hat LSTM ~4× mehr Parameter als ein Vanilla RNN bei gleicher Hidden-Dimension?

**Frage 38:**  
Vergleichen Sie Word2Vec, FastText und BPE:
- Wie geht jeder mit OOV-Wörtern um?
- Welcher erfasst Morphologie am besten?
- Welcher hat die kleinste Vokabulargröße?

**Frage 39:**  
Erklären Sie, warum Transformer langfristige Abhängigkeiten besser modellieren können als LSTM (obwohl LSTM das Vanishing Gradient Problem löst).

**Frage 40:**  
Was ist Teacher Forcing beim Training von Seq2Seq-Modellen? Was ist das Exposure Bias Problem?

---

## Antworten & Lösungen

<details>
<summary>Klicken Sie hier, um die Antworten anzuzeigen</summary>

### Teil A Antworten

**A1:** BoW ignoriert Wortreihenfolge. Beide Sätze haben identische Wortmengen {"Dog", "bites", "man"}. Die syntaktische Struktur (Subjekt-Verb-Objekt) geht verloren.

**A2:**
- TF("cat", D1) = 1/6 ≈ 0.167
- IDF("cat") = log(3/2) ≈ 0.405
- TF-IDF = 0.167 × 0.405 ≈ 0.068

**A3:** IDF("the") = log(3/3) = log(1) = 0. Das Wort trägt keine Diskriminierungskraft bei, da es in allen Dokumenten vorkommt (Stopwort).

**A4:**
| Eigenschaft | BoW/TF-IDF | Word2Vec |
|-------------|------------|----------|
| Dimension | 100.000 | 300 |
| Sparsity | Sparse (>99% Nullen) | Dense (alle Werte ≠ 0) |
| Speicherbedarf | 400 KB | 1.2 KB |
| Semantik | Keine (alle Wörter orthogonal) | Ja (cosine similarity) |

**A5:** "You shall know a word by the company it keeps." Wörter mit ähnlicher Bedeutung treten in ähnlichen Kontexten auf. Beispiel: "Katze" und "Hund" erscheinen beide in Kontexten wie "___ schläft", "___ füttern".

**A6:**
- CBOW: Input=Kontextwörter (gemittelt), Output=Zielwort, Richtung=Kontext→Zielwort, Vorteil=schneller, Nachteil=schlechter für seltene Wörter
- Skip-gram: Input=Zielwort, Output=Kontextwörter, Richtung=Zielwort→Kontext, Vorteil=besser für seltene Wörter, Nachteil=langsamer

**A7:** Word2Vec lernt semantische Beziehungen als Vektor-Operationen. Die Beziehung "König zu Mann" ist ähnlich wie "Königin zu Frau" im Vektorraum. Mathematisch: v(king) - v(man) + v(woman) ≈ v(queen).

**A8:** Negative Sampling: Statt alle |V| Wörter zu betrachten, nur k zufällige negative Beispiele. Reduziert Komplexität von O(|V|) zu O(k). Typisch: k=5-20.

**A9:** FastText zerlegt OOV-Wörter in Character n-grams. Beispiel: "unfriend" → n-grams: <un, unf, nfr, fri, rie, ien, end, nd>. Embedding = Summe der n-gram Embeddings. Teilt n-grams mit bekannten Wörtern.

**A10:**
1. Starte mit allen Characters als Vokabular
2. Zähle alle benachbarten Symbol-Paare im Korpus
3. Finde häufigstes Paar, füge als neues Symbol hinzu
4. Ersetze alle Vorkommen des Paares
5. Wiederhole bis Vokabulargröße erreicht

**A11:** BPE ist Subword-Tokenisierung. Vorteile: Kontrollierbare Vokabulargröße, OOV-Behandlung durch Zerlegung, produktive Morphologie ("unbelievable" → ["un", "believ", "able"]).

**A12:** "ChatGPT" mit n=3: <Ch, Cha, hat, atG, tGP, GPT, PT>

### Teil B Antworten

**A13:** h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
- h_t: Hidden State zum Zeitpunkt t
- h_{t-1}: Vorheriger Hidden State
- x_t: Input zum Zeitpunkt t
- W_h: Hidden-to-Hidden Gewichte (rekurrent)
- W_x: Input-to-Hidden Gewichte
- b_h: Bias

**A14:** h_t hängt von h_{t-1} ab, h_{t-1} von h_{t-2}, usw. Datenabhängigkeitskette: x_0 → h_0 → h_1 → h_2 → ... → h_T. h_{t+1} kann nicht vor h_t berechnet werden.

**A15:** N = d_h × (d_x + d_h + 1) = 256 × (100 + 256 + 1) = 256 × 357 = 91.392 Parameter

**A16:**
- Forget Gate (f_t): Bestimmt was aus Cell State vergessen wird. σ-Ausgabe ∈ (0,1). 0=vergessen, 1=behalten.
- Input Gate (i_t): Bestimmt welche neuen Informationen hinzugefügt werden. σ-Ausgabe + C̃_t (Kandidat, tanh).
- Output Gate (o_t): Bestimmt was aus Cell State als Output ausgegeben wird. σ-Ausgabe.

**A17:**
1. f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
2. i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
3. C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
4. C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
5. o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
6. h_t = o_t ⊙ tanh(C_t)

**A18:** Cell State hat additive Updates (C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t). Wenn f_t ≈ 1 und i_t ≈ 0, fließt Gradient unverändert durch (keine Multiplikation mit W_h bei jedem Schritt). "Constant Error Carousel".

**A19:** N = 4 × d_h × (d_x + d_h + 1) = 4 × 128 × (64 + 128 + 1) = 4 × 128 × 193 = 98.816 Parameter

**A20:** Cell State (C_t): Langzeitgedächtnis, speichert Information über viele Schritte. Hidden State (h_t): Kurzzeitgedächtnis, Output des LSTM.

**A21:** Gates brauchen Werte ∈ (0,1) als "Schalter" → Sigmoid. Cell State Kandidaten brauchen Werte ∈ (-1,1) für ausgewogene Aktivierungen → tanh.

**A22:** GRU hat 2 Gates (Update, Reset), LSTM hat 3 Gates. GRU hat keinen separaten Cell State, nur Hidden State.

**A23:**
1. z_t = σ(W_z · [h_{t-1}, x_t])
2. r_t = σ(W_r · [h_{t-1}, x_t])
3. h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
4. h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

**A24:** Bidirektionales RNN verarbeitet Sequenz in beide Richtungen: Vorwärts (h_t^→) und Rückwärts (h_t^←). Output ist Konkatenation [h_t^→; h_t^←]. Vorteil: Voller Kontext (Vergangenheit + Zukunft). Nachteil: Nicht online-fähig, 2× Parameter.

**A25:** NER: Gesamter Text verfügbar → bidirektional möglich. Autoregressive Generierung: Zukünftige Tokens noch nicht generiert → bidirektional unmöglich.

**A26:** GRU: Weniger Parameter (~75% von LSTM), schnelleres Training, oft vergleichbare Performance. LSTM: Mehr Parameter, "Gold Standard", etwas besser bei sehr langen Abhängigkeiten.

### Teil C Antworten

**A27:** Bottleneck: Gesamter Input-Satz muss in einen einzigen Vektor c = h_T komprimiert werden. Bei langen Sätzen Informationsverlust. c hat feste Dimension (z.B. 512), unabhängig von Satzlänge.

**A28:** Greedy: Wählt immer das wahrscheinlichste nächste Wort. Beam Search: Behält die k besten partiellen Sequenzen (Beams). Vorteil: Bessere Gesamtqualität, findet bessere globale Lösungen.

**A29:** Nach Schritt 1: Beams = {"The" (0.4), "A" (0.3)}. Nach Schritt 2:
- "The cat": 0.4 × 0.5 = 0.20
- "The dog": 0.4 × 0.3 = 0.12
- "A cat": 0.3 × 0.6 = 0.18
- "A dog": 0.3 × 0.2 = 0.06
Behalten: "The cat" (0.20), "A cat" (0.18)

**A30:**
1. Alignment Score berechnen: e_{t,i} = score(s_{t-1}, h_i)
2. Attention-Gewichte: α_{t,i} = softmax(e_{t,i})
3. Context Vector: c_t = Σ_i α_{t,i} · h_i

**A31:** Nein! c_t wird für jeden Decoder-Schritt t neu berechnet. α_{t,i} ändert sich mit t (unterschiedlicher Fokus auf verschiedene Encoder-Positionen).

**A32:** c_t = Σ_i α_{t,i} · h_i, wobei α_{t,i} = softmax(score(s_{t-1}, h_i))

**A33:** score(s_{t-1}, h_i) = v_a^T · tanh(W_a · s_{t-1} + U_a · h_i). Verwendet Decoder-State s_{t-1} und Encoder-State h_i.

**A34:** Klassisch: Nur h_T verfügbar (Bottleneck). Mit Attention: Alle Encoder-States h_i verfügbar, selektiver Zugriff durch α_{t,i}. Kein Informationsverlust.

**A35:** Attention-Matrix (Zeilen=Decoder "The", "dog"; Spalten=Encoder "Der", "Hund"):
- α("The", "Der") ≈ 0.9, α("The", "Hund") ≈ 0.1
- α("dog", "Der") ≈ 0.1, α("dog", "Hund") ≈ 0.9
Diagonale Struktur bei gutem Alignment.

### Teil D Antworten

**A36:**
| Eigenschaft | RNN | LSTM | GRU |
|-------------|-----|------|-----|
| Gates | 0 | 3 | 2 |
| Parameter (d_h=128, d_x=64) | 24.7K | 98.8K | 74.1K |
| Lange Abhängigkeiten | Schlecht | Gut | Gut |
| Training | Schnell | Langsam | Mittel |

**A37:** LSTM hat 4 Gates (f, i, C̃, o), jeder mit W_x und W_h. RNN hat nur 1 Berechnung. Formel: LSTM = 4 × RNN Parameter (ungefähr).

**A38:**
- OOV: Word2Vec=Fehler, FastText=n-grams, BPE=Subword-Zerlegung
- Morphologie: FastText (am besten durch n-grams)
- Kleinste Vokabulargröße: BPE (kontrollierbar, typisch 4K-60K)

**A39:** Transformer: O(1) Pfadlänge zwischen allen Token-Paaren (direkte Attention-Verbindung). LSTM: O(n) Pfadlänge (Information muss durch alle Zwischenschritte fließen). Trotz LSTM's Lösung für Vanishing Gradient: Langsame Informationsausbreitung.

**A40:** Teacher Forcing: Beim Training wird das echte nächste Token als Input verwendet, nicht die eigene Vorhersage. Exposure Bias: Modell wird nur mit echten Daten trainiert, aber bei Inferenz mit eigenen Fehlern konfrontiert → Distributional Shift.

</details>

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 36-40 | 🟢 Exzellent - Bereit für Tag 3 |
| 30-35 | 🟢 Gut - Kleine Wiederholung empfohlen |
| 24-29 | 🟡 Befriedigend - Themen wiederholen |
| 18-23 | 🟡 Ausreichend - Tag 2 wiederholen |
| <18 | 🔴 Nachholbedarf - Zusammenfassung nochmal lesen |

---

## Wichtige Formeln (auswendig lernen!)

**TF-IDF:**
```
TF-IDF(w,d,D) = TF(w,d) × log(|D|/|{d∈D: w∈d}|)
```

**RNN:**
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b_h)
```

**LSTM (6 Gleichungen):**
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
N = 4 × d_h × (d_x + d_h + 1)
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

---

**Viel Erfolg!** 🎯

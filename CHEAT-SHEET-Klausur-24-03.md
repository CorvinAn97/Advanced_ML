# 🎯 Advanced ML Cheat Sheet – Klausur 24.03.2026
**Doppelseitig DIN A4 | Priorisiert nach Klausurrelevanz**

---

## 📄 SEITE 1: AUSFÜHRLICHE KONZEPTE (Embeddings, LSTM, Transformer)

### A) WORD EMBEDDINGS (Ausführlich)

#### **Distributional Hypothesis (Firth 1957)**
**Kernidee:** "Ein Wort ist bekannt durch die Gesellschaft, in der es erscheint." Wörter mit ähnlichen Kontexten haben ähnliche Bedeutungen. Diese Hypothese begründet das gesamte Feld der distributiven Semantik und ist die theoretische Basis für Word2Vec, GloVe und moderne Embeddings.

**Mathematische Konsequenz:** Wenn Wörter w1 und w2 in ähnlichen Kontexten c erscheinen, dann sollten ihre Vektorrepräsentationen v(w1) ähnlich v(w2) sein. Dies wird durch kosinische Ähnlichkeit gemessen: Das Skalarprodukt der Vektoren dividiert durch das Produkt ihrer Normen.

**Beispiel:** "Katze" und "Hund" erscheinen beide in Kontexten wie "das ___ bellt", "mein ___", "ein süßer ___" → ihre Embeddings sind ähnlich.

---

#### **Word2Vec: CBOW vs Skip-gram (Im Detail)**

**CBOW (Continuous Bag-of-Words):**
- **Richtung:** Kontext → Zielwort
- **Architektur:** Mehrere Kontextwörter (z.B. ±2 Wörter) werden als Input gemittelt, dann durch Hidden Layer, dann Softmax über Vokabular zur Vorhersage des Zielworts
- **Funktionsweise:** Der Kontext wird zusammengefasst (gemittelt oder summiert), wodurch Wortorder-Information verloren geht, aber das Training stabiler und schneller wird
- **Vorteile:** Schnellere Training (mittelt Kontext, weniger Varianz), gut für häufige Wörter
- **Nachteile:** Verliert Wortordnung, schlechter für seltene Wörter

**Skip-gram:**
- **Richtung:** Zielwort → Kontext
- **Architektur:** Einzelnes Zielwort als Input, vorhersagt mehrere Kontextwörter
- **Funktionsweise:** Für ein Zielwort werden mehrere Kontextwörter vorhergesagt, was mehrfache Updates pro Schritt erfordert
- **Vorteile:** Bessere Qualität für seltene Wörter, erfasst feinere semantische Nuancen
- **Nachteile:** Langsamer (mehrfache Vorhersagen pro Schritt), höhere Varianz

**Negative Sampling:** Statt vollem Softmax (alle Vokabularwörter) nur k negative Beispiele plus ein positives. Dies reduziert die Rechenkomplexität drastisch und macht Training auf großen Vokabularen erst praktikabel.

---

#### **FastText (Subword n-grams, OOV-Handling)**

**Kerninnovation:** Wörter werden als Summe ihrer Character n-gram Vektoren repräsentiert, nicht als atomare Einheiten.

**Mechanismus:**
1. Für Wort w mit Länge m: Extrahiere alle n-grams der Länge 3-6 (inkl. Special-Tokens für Wortbegin und -ende)
2. Beispiel "where": Extrahiere Bigramme und Trigramme wie "wh", "whe", "wher", "here", "ere", "re" plus Specials
3. Wortvektor: Summe aller n-gram Vektoren

**OOV-Handling (Out-of-Vocabulary):** Unbekannte Wörter können aus ihren Subwords komponiert werden. "unhappiness" war nicht im Training, aber "un-", "happy", "-ness" waren es → Vektor ist komponierbar.

**Vorteile:**
- Morphologische Generalisierung (z.B. "laufen", "läuft", "gelaufen" teilen n-grams)
- Robustheit gegenüber Rechtschreibfehlern
- Funktioniert für alle Wörter, auch OOV

**Nachteile:**
- Größeres Vokabular (n-grams statt Wörter)
- Langsamere Inferenz (mehr Lookups)

---

#### **BPE (Byte Pair Encoding) – Vokabularaufbau**

**Algorithmus:**
1. **Initialisierung:** Startvokabular = alle einzelnen Zeichen (Characters)
2. **Iteration:** Finde häufigstes Bigramm im Korpus, fusioniere zu neuem Symbol
3. **Wiederholung:** Repeat bis Vokabulargröße erreicht (z.B. 30.000-50.000)

**Beispiel:** "hello" + "world"
- Start: h,e,l,l,o,w,o,r,l,d
- Schritt 1: "ll" ist häufig → neue Symbol "ll"
- Schritt 2: "he" ist häufig → "he"
- Schritt 3: "ll" + "o" → "llo"
- Ergebnis: "hello" = h,e,llo; "world" = w,o,r,l,d

**Vokabulargröße Trade-off:**
- Klein (5.000): Viele unbekannte Wörter, viele OOV
- Mittel (30.000): Guter Kompromiss
- Groß (100.000): Weniger OOV, aber größere Embedding-Matrix

**WordPiece (Google):** Ähnlich BPE, aber fusioniert nur wenn das neue Token die Likelihood des Trainingskorpus erhöht (nicht nur Frequenz).

---

#### **TF-IDF (Term Frequency, IDF, Anwendung)**

**Term Frequency (TF):** Lokale Wichtigkeit eines Wortes in einem Dokument.
- Berechnet als Häufigkeit des Wortes im Dokument dividiert durch Gesamtwortzahl
- Alternative: logarithmische Dämpfung häufiger Wörter

**Inverse Document Frequency (IDF):** Globale Seltenheit eines Wortes im Korpus.
- Berechnet als Logarithmus des Verhältnisses: Gesamtzahl Dokumente durch Dokumente die das Wort enthalten
- Seltene Wörter (niedriger document frequency) → hoher IDF

**TF-IDF:** Produkt aus TF und IDF
- Hohe TF-IDF: Wort ist häufig in diesem Dokument, aber selten insgesamt → guter Diskriminator
- Anwendung: Dokumentenähnlichkeit, Suchmaschinen-Ranking, Feature-Extraction für ML

**Beispiel:** "Der" hat hohe TF in allen Dokumenten, aber auch hohen document frequency → niedriger IDF → niedrige TF-IDF. "NeuronalesNetz" hat niedrigen document frequency → hoher IDF → hohe TF-IDF wenn es vorkommt.

---

#### **Bias in Embeddings (Caliskan et al. 2017)**

**Studie:** "Semantics derived automatically from language corpora contain human-like biases" (Science 2017)

**Feststellung:** Word2Vec-Embeddings aus News-Korpora zeigen:
- Geschlechterstereotype: "Mann" assoziiert mit "Programmierer", "Frau" assoziiert mit "Haushalt"
- Rassistische Assoziationen: Europäische Namen positiver konnotiert als Afrikanische Namen
- Berufliche Zuordnungen: "Arzt" näher an "Mann", "Krankenschwester" näher an "Frau"

**WEAT (Word Embedding Association Test):** Misst Bias analog zum IAT (Implicit Association Test).
- Teststatistik vergleicht Assoziationsstärken zwischen Zielgruppen und Attributen
- Signifikanter Effekt zeigt systematische Verzerrung

**Implikation:** ML-Systeme, die auf solchen Embeddings basieren, perpetuieren gesellschaftliche Vorurteile. Gegenmaßnahmen: Debiasing (Hard/Soft Debiasing), faire Regularisierung, bewusste Datenselektion.

---

### B) LSTM / RNN (Ausführlich)

#### **Vanilla RNN und Vanishing Gradient Problem**

**Vanilla RNN:** Hidden State kodiert gesamte Vergangenheit, sequentiell verarbeitet.
- Hidden State hängt von vorherigem Hidden State und aktueller Input ab
- Sequentiell: Jeder Schritt braucht Ergebnis des vorherigen → keine Parallelisierung

**Vanishing Gradient – Mathematische Analyse:**
Bei Backpropagation Through Time (BPTT) werden Gradienten über die Zeit zurückpropagiert.
- Gradient an frühem Zeitpunkt ist Produkt vieler partieller Ableitungen
- Jede Ableitung enthält die Ableitung der Aktivierungsfunktion (tanh oder sigmoid)
- tanh und sigmoid haben Ableitungen kleiner oder gleich eins, typisch deutlich kleiner (circa 0.25 für große Eingaben)
- Nach vielen Schritten: Produkt vieler Zahlen < 1 → Gradient wird exponentiell klein

**Konsequenz:** Gradienten für frühe Zeitpunkte verschwinden → lange Abhängigkeiten werden nicht gelernt. Das Netzwerk kann Information über lange Distanzen nicht effektiv propagieren.

**LSTM-Lösung:** Expliziter Cell State mit additiver Verbindung (nicht multiplikativ!) → Gradient kann unverändert fließen.

---

#### **LSTM Gates im Detail (Forget, Input, Output)**

**Forget Gate:** Entscheidet, welche Information aus dem Cell State gelöscht wird.
- Kombiniert vorherigen Hidden State und aktuellen Input durch lineare Transformation
- Sigmoid-Aktivierung erzeugt Werte zwischen 0 und 1
- Wert nahe 0: Information löschen (vergessen)
- Wert nahe 1: Information behalten (erinnern)
- **Beispiel:** Bei Satzende (Punkt) → Forget Gate hoch für Satzgedächtnis, niedrig für temporäre Details

**Input Gate:** Entscheidet, welche neue Information zum Cell State hinzugefügt wird.
- Ähnliche Struktur wie Forget Gate
- Steuert den Zufluss von neuem Inhalt
- Arbeitet zusammen mit Candidate Cell State

**Candidate Cell State:** Vorschlag für neue Information.
- Lineare Transformation von Input und vorherigem Hidden State
- tanh-Aktivierung → Wertebereich -1 bis 1, kann Information hinzufügen oder subtrahieren

**Output Gate:** Entscheidet, welche Information aus dem Cell State zum Hidden State wird.
- Filtert den Cell State durch tanh und multipliziert mit Output Gate
- Ergebnis ist der neue Hidden State (wird an nächste Schicht und nächsten Zeitpunkt weitergegeben)

**Cell State Update:**
- Additive Struktur: alter Cell State (gewichtet mit Forget Gate) plus neue Information (gewichtet mit Input Gate und Candidate)
- Additive Struktur ermöglicht direkten Gradientenfluss über viele Zeitschritte
- Wenn Forget Gate nahe 1: Gradient fließt unverändert → kein Vanishing Gradient

---

#### **Peephole Connections**

**Standard-LSTM:** Gates hängen nur von vorherigem Hidden State und aktuellem Input ab.
**Peephole-LSTM:** Gates haben zusätzlich direkten Zugriff auf den Cell State.

**Modifizierte Gates:**
- Forget Gate, Input Gate, und Output Gate erhalten zusätzlich den Cell State als Input
- Output Gate greift auf aktuellen Cell State zu (nicht vorherigen)

**Vorteil:** Gates können explizit auf den Cell State reagieren (z.B. "wenn Cell State groß, dann Output hoch"). Empirisch bessere Performance auf manchen Tasks, insbesondere Timing-kritischen Aufgaben.

---

#### **Coupled Gates**

**Idee:** Forget und Input Gates sind gekoppelt – was vergessen wird, wird durch neue Information ersetzt.
- Forget Gate ist Komplement des Input Gates (Summe = 1)
- Nur Input Gate wird gelernt, Forget Gate ist abgeleitet

**Vorteil:** Weniger Parameter, stabileres Training (kein "doppelter Entscheid")
**Nachteil:** Weniger Flexibilität (Forget und Input sind nicht unabhängig)

---

#### **Bidirectional LSTM**

**Problem:** Unidirektionale LSTM kennt nur linken Kontext (Vergangenheit).
**Lösung:** Zwei LSTMs parallel:
- **Forward-LSTM:** Verarbeitet Sequenz von Anfang bis Ende (linker Kontext)
- **Backward-LSTM:** Verarbeitet Sequenz von Ende bis Anfang (rechter Kontext)
- **Output:** Konkatenation beider Hidden States (doppelte Dimension)

**Anwendung:** Sequence Labeling (NER, POS-Tagging), wo gesamter Kontext verfügbar ist.
**Nicht geeignet:** Autoregressive Generation (Decoder), wo nur Vergangenheit bekannt sein darf.

---

#### **Parameter Count Berechnung**

**Pro LSTM-Layer:**
- 4 Gates (Forget, Input, Candidate, Output)
- Pro Gate: Input-Gewicht, Hidden-Gewicht (rekurrent), und Bias
- Input-Gewicht: Input-Dimension mal Hidden-Dimension
- Hidden-Gewicht: Hidden-Dimension mal Hidden-Dimension (rekurrent)
- Bias: Hidden-Dimension pro Gate

**Formel:** 4 mal [(Input-Dimension mal Hidden-Dimension) plus (Hidden-Dimension mal Hidden-Dimension) plus Hidden-Dimension]

**Beispiel:** Input 64, Hidden 128
- Input-Gewicht pro Gate: 64 mal 128 = 8192
- Hidden-Gewicht pro Gate: 128 mal 128 = 16384
- Bias pro Gate: 128
- Pro Gate: 8192 plus 16384 plus 128 = 24704
- Gesamt: 4 mal 24704 = 98816 Parameter

**Vergleich GRU:** 3 Gates statt 4 → etwa 75% der LSTM-Parameter.

---

### C) TRANSFORMER (Ausführlich)

#### **Self-Attention Mechanismus (Q, K, V im Detail)**

**Kernidee:** Jedes Token attendiert auf alle Tokens (inkl. sich selbst) gewichtet nach Relevanz.

**Berechnung:**
1. **Input:** Token-Embedding (kontinuierlicher Vektor)
2. **Projektionen:** Drei lineare Transformationen erzeugen Query, Key, und Value Vektoren
   - Query: "Was suche ich?" (aktuelles Token fragt nach relevanten Informationen)
   - Key: "Was biete ich an?" (Token signalisiert seine Relevanz für Queries)
   - Value: "Welche Information trage ich?" (tatsächlicher Inhalt)
3. **Attention-Score:** Dot-Product zwischen Query und Key (Misst Ähnlichkeit/Relevanz)
4. **Attention-Weights:** Softmax über alle Scores (normalisiert zu Wahrscheinlichkeiten)
5. **Output:** Gewichtete Summe aller Value-Vektoren

**Matrix-Form:** Attention-Matrix multipliziert mit Value-Matrix ergibt gewichtete Kombination.

**Interpretation:**
- Query sucht relevante Keys (hoher Dot-Product = hohe Relevanz)
- Value enthält die eigentliche Information
- Output ist gewichteter Durchschnitt aller Values, gewichtet nach Relevanz

**Beispiel:** "The animal didn't cross the street because it was too tired"
- "it" attendiert stark auf "animal" (Query-Key Ähnlichkeit hoch)
- "it" attendiert schwach auf "street"
- → Coreference Resolution durch Attention

---

#### **Scaled Dot-Product Attention (Warum Skalierung?)**

**Problem:** Ohne Skalierung werden Dot-Products bei großer Dimension extrem groß.
- Dot-Product summiert viele Multiplikationen
- Bei großen Dimensionen: Varianz des Dot-Products wächst mit Dimension
- Standardabweichung: Wurzel der Dimension

**Folge:** Dot-Products haben große Beträge ohne Skalierung.

**Warum Skalierung wichtig?**
- Softmax hat Gradienten, die bei großen Eingabebeträgen verschwinden
- Bei großen Beträgen: Softmax-Ausgang nahe 0 oder 1 → Gradient nahe 0 (Vanishing Gradient!)
- Skalierung hält Eingaben im linearen Bereich von Softmax

**Effekt:** Stabileres Training, bessere Gradientenfluss.

---

#### **Multi-Head Attention (Wie funktioniert Concat?)**

**Idee:** Mehrere Attention-Heads parallel lernen verschiedene Beziehungen.

**Mechanismus:**
1. **Mehrere Heads:** Für jeden Head separate Projektionsmatrizen für Query, Key, Value
2. **Parallele Berechnung:** Alle Heads unabhängig berechenbar (Parallelisierung!)
3. **Concat:** Alle Head-Outputs konkateniert (aneinandergehängt)
4. **Output-Projektion:** Lineare Transformation des konkatenierten Vektors

**Typische Werte:** 8 Heads (Base), 16 Heads (Large), Dimension pro Head = Gesamt-Dimension / Anzahl Heads

**Warum Multi-Head?**
- Verschiedene Heads lernen verschiedene Muster
- Head 1: Syntaktische Beziehungen (Subjekt-Verb)
- Head 2: Coreference (Pronomen → Nomen)
- Head 3: Semantische Ähnlichkeit
- Head 4: Positionsabhängigkeiten
- **Empirisch:** Verschiedene Heads lernen verschiedene Aspekte der Repräsentation

---

#### **Positional Encoding (Sinus/Cosinus vs RoPE)**

**Problem:** Self-Attention ist positionsinvariant (permutationsinvariant). Ohne Positionsinfo ist "The cat bit the dog" gleich wie "The dog bit the cat".

**Original (Vaswani et al. 2017):**
- Sinus und Cosinus Funktionen mit unterschiedlichen Frequenzen
- Niedrigere Dimensionen: höhere Frequenz (schnelle Oszillation)
- Höhere Dimensionen: niedrigere Frequenz (langsame Oszillation)
- Addition zu Token-Embeddings

**Eigenschaften:**
- Unterschiedliche Frequenzen pro Dimension ermöglichen Extrapolation (längere Sequenzen als Training)
- Relationale Information: Positionsverschiebung ist linear transformierbar

**RoPE (Rotary Position Embedding, modern):**
- Rotiert Query und Key Vektoren in 2D-Ebenen
- Nur für Query und Key, nicht Value
- **Vorteil:** Bessere Extrapolation, einfacher zu implementieren, SOTA in LLaMA, PaLM

---

#### **Encoder vs Decoder (Unterschiede, Cross-Attention)**

**Encoder (Bidirektional):**
- **Attention:** Nur Self-Attention (alle Tokens attendieren auf alle)
- **Masking:** Keine Maskierung (volles Kontextfenster)
- **Anwendung:** Understanding-Tasks (Klassifikation, NER, BERT)
- **Output:** Kontextuelle Repräsentation aller Tokens

**Decoder (Kausal / Autoregressiv):**
- **Attention 1:** Masked Self-Attention (nur auf Vergangenheit)
- **Attention 2:** Cross-Attention (Key und Value vom Encoder, Query vom Decoder)
- **Masking:** Causal Mask (untere Dreiecksmatrix, minus unendlich für zukünftige Tokens)
- **Anwendung:** Generation-Tasks (Translation, Text-Generation, GPT)
- **Output:** Nächstes Token (autoregressiv)

**Cross-Attention:**
- Query aus Decoder (aktuelles Decoder-Token)
- Key und Value aus Encoder (alle Encoder-Tokens)
- **Funktion:** Decoder attendiert auf relevante Encoder-Informationen
- **Beispiel:** Translation: Decoder-Token "the" attendiert auf Encoder-Token "der"

---

#### **Masked Attention (Causal Masking)**

**Zweck:** Verhindert "Cheating" – Decoder darf nur auf Vergangenheit attendieren.

**Implementierung:**
- Mask-Matrix: Token i darf auf Token j attendieren nur wenn i größer gleich j
- Für zukünftige Tokens: Wert minus unendlich setzen vor Softmax
- Softmax von minus unendlich ist null → zukünftige Tokens haben Gewicht null

**Effekt:** Jedes Token sieht nur sich selbst und vorherige Tokens.

**Training:** Alle Tokens parallel berechenbar (Teacher Forcing).
**Inferenz:** Autoregressiv, ein Token nach dem anderen.

---

#### **LayerNorm (Pre vs Post)**

**Layer Normalization:**
- Normalisiert über Feature-Dimensionen (nicht Batch!)
- Berechnet Mittelwert und Varianz über Features
- Normalisiert zu Mittelwert 0, Varianz 1
- Lernbare Parameter (Skalierung und Verschiebung) für Flexibilität

**Pre-Norm (modern):**
- Normalisierung vor Sub-Layer
- **Vorteile:** Stabileres Training, bessere Gradienten-Flow, ermöglicht tiefere Modelle
- **Nachteil:** Etwas schlechtere Training-Effizienz (frühe Schichten lernen langsamer)

**Post-Norm (original Transformer):**
- Normalisierung nach Sub-Layer (nach Addition)
- **Vorteile:** Bessere finale Performance bei kleinen Modellen
- **Nachteil:** Instabil bei tiefen Modellen, Gradienten-Probleme

**SOTA:** Pre-Norm ist Standard in modernen Architekturen (LLaMA, etc.)

---

#### **FFN (SwiGLU vs GELU)**

**Feed-Forward Network:** Position-wise FFN nach jeder Attention-Schicht.

**Original (GELU/ReLU):**
- Lineare Transformation, Aktivierung (GELU oder ReLU), lineare Transformation
- Dimension typisch 4 mal Modell-Dimension

**SwiGLU (SOTA, gated):**
- Zwei parallele lineare Projektionen
- Eine Projektion durch Swish-Aktivierung (x mal Sigmoid von x)
- Elementweise Multiplikation der beiden Pfade (Gating)
- Output-Projektion

**Vorteil:** Gating-Mechanismus (ähnlich LSTM Gates) ermöglicht selektive Informationsweitergabe.

**Dimension:** Etwa 2/3 der originalen FFN-Dimension bei gleicher Parameterzahl.

**Warum SwiGLU besser?**
- Empirisch bessere Performance auf LLM-Benchmarks
- Verwendet in PaLM, LLaMA, Chinchilla

---

## 📄 SEITE 2: AUSFÜHRLICHE KONZEPTE (RL, Generative Modelle, XAI, IL) + Fragen + Hinweise

### D) REINFORCEMENT LEARNING (Ausführlich)

#### **Q-Learning (Off-Policy, Update-Regel)**

**Kernidee:** Lerne Q-Funktion, die den erwarteten Return bei Aktion a in Zustand s schätzt, dann optimale Policy danach.

**Update-Regel:**
- Q-Wert wird aktualisiert in Richtung des TD-Targets
- TD-Target: Immediate Reward plus diskontierter maximaler Q-Wert des nächsten Zustands
- Lernrate kontrolliert Schrittweite der Aktualisierung
- Diskontfaktor gewichtet zukünftige Rewards

**Komponenten:**
- Lernrate: Schrittweite (klein = stabil, groß = schnell)
- Immediate Reward: Direkte Belohnung der Aktion
- Diskontfaktor: Gewichtung zukünftiger Rewards (nahe 1 = weitblickend, nahe 0 = kurzsichtig)
- TD-Target: Optimistische Schätzung des nächsten Werts (maximaler Q-Wert)
- TD-Error: Differenz zwischen Target und aktuellem Q-Wert (Lernsignal)

**Off-Policy:** Lernt optimale Policy unabhängig von der Behavior-Policy (zum Beispiel epsilon-greedy Exploration). Die gelernte Policy ist die beste, auch wenn während des Trainings explorative Aktionen gewählt werden.

**Konvergenz:** Garantiert unter GLIE (Greedy in Limit with Infinite Exploration):
- Exploration geht gegen null mit Zeit gegen unendlich
- Alle Zustand-Aktions-Paare werden unendlich oft besucht

**Beispiel:** Grid-World
- Zustand: (x,y)-Position
- Aktion: hoch, runter, links, rechts
- Reward: +1 bei Ziel, -0.01 pro Schritt
- Q-Learning lernt kürzesten Pfad

---

#### **Double DQN (Overestimation Problem)**

**Problem:** Maximum-Operator überschätzt Q-Werte systematisch.
- Erwartungswert des Maximums ist größer als Maximum der Erwartungswerte (Jensen-Ungleichung)
- Noise in Q-Schätzung wird als Signal interpretiert
- Systematische Überschätzung führt zu suboptimaler Policy

**Double DQN-Lösung:** Entkopplung von Selektion und Bewertung.
1. **Selektion:** Beste Aktion wird mit Q-Netzwerk gewählt
2. **Bewertung:** Wert der besten Aktion wird mit Target-Netzwerk bewertet

**Effekt:** Noise zwischen Selektion und Bewertung ist unkorreliert → Overestimation reduziert.

---

#### **SARSA (On-Policy, Unterschied zu Q-Learning)**

**Update-Regel:**
- Q-Wert wird aktualisiert in Richtung des tatsächlichen nächsten Q-Werts
- Tatsächlicher nächster Q-Wert: Reward plus diskontierter Q-Wert der tatsächlich gewählten nächsten Aktion

**Unterschied zu Q-Learning:**
- Q-Learning: maximaler Q-Wert des nächsten Zustands (off-policy, optimistisch)
- SARSA: Q-Wert der tatsächlichen nächsten Aktion (on-policy)

**Behavior-Policy:** Nächste Aktion wird von aktueller Policy gewählt (zum Beispiel epsilon-greedy).

**Konsequenz:**
- SARSA lernt die Policy, die es tatsächlich ausführt (inkl. Exploration)
- Vorsichtiger bei riskanten Zuständen (berücksichtigt epsilon-greedy Fehler)
- Q-Learning lernt optimale Policy, ignoriert Exploration-Risiko

**Beispiel:** Cliff-Walking
- Q-Learning: Lernt optimalen Pfad (entlang Cliff), aber fällt oft runter (epsilon-greedy Fehler)
- SARSA: Lernt sicheren Pfad (weg vom Cliff), berücksichtigt epsilon-Fehler

---

#### **Experience Replay**

**Idee:** Speichere Transitions (Zustand, Aktion, Reward, nächster Zustand) in Buffer, sample Mini-Batches uniform.

**Vorteile:**
1. **Dekorrelation:** Sequentielle Samples sind korreliert (Markov-Kette). Uniform-Sampling bricht Korrelation.
2. **Effizienz:** Jede Transition kann mehrfach verwendet werden (höhere Sample-Effizienz).
3. **Stabilität:** Vermeidet "catastrophic forgetting" von seltenen, wichtigen Transitions.

**Implementierung:**
- Buffer-Größe: 10.000 bis 1.000.000 Transitions
- Mini-Batch-Größe: 32 bis 512
- Sampling: Uniform aus Buffer

**Prioritized Experience Replay:** Sample wichtige Transitions häufiger (hoher TD-Error → wichtiger Lernsignal).

---

#### **Target Networks**

**Problem:** Q-Werte ändern sich bei jedem Update → TD-Target "bewegt sich" (Moving Target).

**Lösung:** Separates Target-Network mit verzögerten Gewichten.

**Update-Strategien:**
1. **Periodic Copy:** Alle C Schritte: Target-Gewichte werden mit Q-Gewichten kopiert
2. **Polyak-Averaging:** Target-Gewichte werden als gewichteter Durchschnitt von alten Target-Gewichten und Q-Gewichten aktualisiert (tau klein, soft update)

**Effekt:** TD-Target ist stabil über mehrere Schritte → konvergentes Training.

---

#### **UCB (Optimismus bei Unsicherheit)**

**Idee:** Wähle Aktion mit höchster oberer Konfidenzgrenze.

**Formel:**
- Geschätzter Mean-Reward plus Bonus für Unsicherheit
- Bonus: Wurzel aus (2 mal Logarithmus der Zeit dividiert durch Anzahl Besuche der Aktion)

**Komponenten:**
- Geschätzter Mean-Reward für Aktion
- Anzahl Besuche der Aktion
- Bonus für Unsicherheit

**Eigenschaften:**
- Bonus sinkt mit Anzahl Besuchen: Mehr Besuche → weniger Unsicherheit
- Bonus steigt mit Logarithmus der Zeit: Späte Zeitpunkte → höhere Exploration für unbesuchte Aktionen
- **Regret:** Wurzel aus (Anzahl Aktionen mal Zeit mal Logarithmus der Zeit) (optimal für Bandits)

**Intuition:** "Optimismus bei Unsicherheit" – wenn du unsicher bist, nimm das Beste an.

---

#### **Thompson Sampling**

**Idee:** Sample aus Posterior-Verteilung über Parameter, wähle optimale Aktion unter Sample.

**Algorithmus:**
1. Maintain Posterior-Verteilung über Reward-Parameter
2. Sample Parameter aus Posterior
3. Wähle Aktion mit höchstem erwarteten Reward unter gesampletem Parameter
4. Observiere Reward, update Posterior

**Implementation (Beta-Binomial):**
- Prior: Beta-Verteilung über Reward-Wahrscheinlichkeit
- Update: Parameter erhöhen bei Success, andere bei Failure
- Sample: Beta-Verteilung sample

**Eigenschaften:**
- Automatische Balance Exploration/Exploitation
- Empirisch sehr stark (oft besser als UCB)
- Theoretisch schwer zu analysieren (aber gute Regret-Bounds bewiesen)

---

### E) GENERATIVE MODELLE (Ausführlich)

#### **GAN (Generator, Discriminator, Minimax, Mode Collapse)**

**Architektur:**
- **Generator:** Noise → Fake Sample (erzeugt synthetische Daten)
- **Discriminator:** Eingabe → Wahrscheinlichkeit (Real vs Fake)

**Minimax-Game:**
- Discriminator maximiert: Echte Samples korrekt klassifizieren, Fake-Samples als Fake erkennen
- Generator minimiert: Discriminator soll Fake-Samples für echt halten
- Gegenspielerisches Training (Adversarial)

**Training:**
1. Fixiere Generator, trainiere Discriminator (maximiere Discriminator-Loss)
2. Fixiere Discriminator, trainiere Generator (minimiere Generator-Loss)

**Mode Collapse:**
- **Problem:** Generator produziert nur 1 Mode (z.B. nur "6" bei MNIST)
- **Ursache:** Discriminator wird zu gut, Gradient für Generator verschwindet (Vanishing Gradient)
- **Symptom:** Generator lernt "sichere" Fake-Samples, die Discriminator täuschen, aber keine Vielfalt

**Lösungen:**
- **Wasserstein GAN:** Critic statt Discriminator, Earth Mover's Distance
- **Gradient Penalty:** Lipschitz-Constraint erzwingen (Gradient-Norm begrenzen)
- **Mini-Batch Discrimination:** Discriminator sieht Batch-Statistiken, erkennt Mode Collapse
- **Experience Replay:** Buffer von Fake-Samples, Discriminator trainiert auf Historie

---

#### **VAE (Encoder, Decoder, Reparametrisierung, ELBO)**

**Architektur:**
- **Encoder:** Input → Parameter der Posterior (Mittelwert und Varianz)
- **Decoder:** Latent Variable → Rekonstruktion

**Latent Variable:** Sample aus Posterior-Verteilung (Gaussian mit Encoder-Parametern)

**Reparametrisierungs-Trick:**
- Problem: Sampling ist nicht differenzierbar (kein Gradient durch Sampling)
- Lösung: Sample als deterministische Funktion von Noise
- Noise wird extern gesample, Gradient kann durch Parameter fließen

**ELBO (Evidence Lower Bound):**
- Reconstruction: Erwartete Log-Likelihood der Rekonstruktion
- KL-Regularization: Divergenz zwischen Posterior und Prior (Gaussian)
- Trade-off: Hohe Reconstruction vs strukturierter Latent Space

**Trade-off:**
- Hoher Reconstruction: Rekonstruktion gut, aber Posterior ungleich Prior (kein strukturierter Latent Space)
- Hoher KL: Posterior ungefähr Prior, aber Rekonstruktion schlecht (Posterior Collapse)
- **Beta-VAE:** KL stärker gewichten für disentanglement

---

#### **Diffusion (Forward/Reverse Process, Classifier-Free Guidance)**

**Forward Process:**
- Schrittweise Noise-Addition über viele Schritte
- Jeder Schritt: Little Noise addieren
- Endzustand: Reines Rauschen (Gaussian)
- **Closed Form:** Direkte Berechnung jedes Steps möglich (Linearkombination von Original und Noise)

**Reverse Process:**
- Neuronales Netzwerk lernt Noise vorherzusagen
- Umkehrung des Forward-Prozesses: Noise schrittweise entfernen
- Training: Netzwerk minimiert Fehler in Noise-Vorhersage

**Sampling:**
1. Sample reines Rauschen
2. Iterativ: Noise entfernen (Reverse-Step)
3. Output: Generiertes Bild

**Classifier-Free Guidance:**
- **Idee:** Condition (z.B. Text) ohne separaten Classifier
- **Training:** Randomly drop Condition (unconditional) mit Wahrscheinlichkeit
- **Inferenz:** Linearkombination von unconditional und conditional Prediction
  - Gewicht 1: Standard conditional
  - Gewicht größer 1: Stronger guidance (schärfer, aber weniger divers)
  - Gewicht 0: Unconditional

---

### F) XAI (Ausführlich)

#### **LIME (Lokale Approximation, Surrogate Model)**

**Idee:** Approximiere komplexes Modell lokal durch interpretierbares Surrogate-Modell.

**Algorithmus:**
1. Wähle zu erklärende Instanz
2. Generiere Perturbationen: Kleine Änderungen der Eingabe
3. Hole Vorhersagen vom Originalmodell für Perturbationen
4. Gewichte: Lokale Gewichtung (nahe Instanz = höheres Gewicht)
5. Trainiere Surrogate (linear): Minimiert gewichteten Fehler
6. Erklärung: Lineare Koeffizienten des Surrogate

**Eigenschaften:**
- **Lokal:** Gültig nur nahe der gewählten Instanz
- **Modellunabhängig:** Black-Box-fähig (jedes Modell erklärbar)
- **Surrogate:** Linear, Entscheidungsbaum, etc.

**Nachteil:** Instabil (Perturbationen zufällig), keine theoretische Garantie.

---

#### **SHAP (Shapley Values, fairer Beitrag)**

**Theorie:** Shapley Values aus kooperativer Spieltheorie – faire Aufteilung des "Gewinns" (Vorhersage) auf "Spieler" (Features).

**Shapley Value:**
- Summe über alle möglichen Feature-Subsets (ohne das betrachtete Feature)
- Für jedes Subset: Marginaler Beitrag des Features (Vorhersage mit Feature minus Vorhersage ohne Feature)
- Gewichtung: Kombinatorischer Faktor (faoore Aufteilung)

**Komponenten:**
- Menge aller Features
- Subset von Features (Koalition ohne das betrachtete Feature)
- Wert der Koalition (Vorhersage mit Features)
- Marginaler Beitrag: Unterschied mit/ohne Feature
- Gewichtung: Fairer kombinatorischer Faktor

**Eigenschaften:**
- **Effizienz:** Summe aller Shapley Values = Vorhersage minus Baseline (vollständige Zurechnung)
- **Symmetrie:** Gleiche Beiträge → gleiche Shapley Values
- **Null-Feature:** Irrelevantes Feature → Shapley Value 0
- **Additivität:** Shapley Values addieren sich bei Summe von Funktionen

**Berechnung:** Exakt exponentiell (alle Subsets). Approximationen:
- **KernelSHAP:** LIME mit Shapley-Kernel
- **DeepSHAP:** Backprop-Regeln für Neuronalen Netze
- **TreeSHAP:** Polynomial für Bäume

---

#### **PFI (Permutation Feature Importance)**

**Idee:** Feature ist wichtig, wenn Permutation die Accuracy verschlechtert.

**Algorithmus:**
1. Trainiere Modell, messe Baseline-Accuracy
2. Für jedes Feature:
   - Permutiere Feature (zufällige Reihenfolge der Werte)
   - Messe Accuracy mit permutiertem Feature
   - Wichtigkeit: Baseline-Accuracy minus Accuracy nach Permutation
3. Hohe Wichtigkeit → wichtiges Feature

**Eigenschaften:**
- **Global:** Über gesamten Datensatz
- **Modellunabhängig:** Black-Box-fähig
- **Einfach:** Nur Accuracy-Messung nötig

**Problem bei korrelierten Features:**
- Wenn zwei Features korreliert: Permutation eines hat kleinen Effekt (anderes kompensiert)
- Folge: Beide Features erscheinen unwichtig
- **Lösung:** Clustere Features, behalte eines pro Cluster

---

### G) IMITATION LEARNING

#### **Behavioral Cloning (Distributional Shift)**

**Idee:** Trainiere Policy via Supervised Learning auf Experten-Demonstrationen.

**Problem: Distributional Shift:**
- Training: Experten-Trajectorien
- Test: Policy-Trajectorien
- **Folge:** Fehler akkumulieren sich → Agent erreicht Zustände außerhalb Trainingsdaten

**Theorie:**
- i.i.d. Annahme: Lineare Fehlerakkumulation (Fehlerrate mal Horizont)
- Distributional Shift: Quadratische Fehlerakkumulation (Fehlerrate mal Horizont quadratisch)

**Beispiel:** Autonomes Fahren
- Experte: Fährt immer in der Spur
- Policy: Kleiner Fehler → Spur verlassen
- Trainingsdaten: Nur "in Spur"-Zustände
- Test: "außerhalb Spur"-Zustände → Policy unsicher → größerer Fehler

---

#### **DAgger (Algorithmus, wie löst es das Problem?)**

**Dataset Aggregation (DAgger):** Iterative Datensammlung mit Policy-Rollouts.

**Algorithmus:**
1. Trainiere initiale Policy auf initialen Experten-Daten
2. Iteration:
   - **Rollout:** Sammle Trajektorien unter aktueller Policy
   - **Labeling:** Experte labelt gesammelte Zustände mit korrekten Aktionen
   - **Aggregation:** Alte Daten plus neue gelabelte Daten
   - **Training:** Neue Policy auf aggregierten Daten
3. Output: Finale Policy

**Warum es funktioniert:**
- Neue Daten enthalten Zustände, die die Policy tatsächlich besucht
- Experte labelt diese Zustände → Trainingsdaten decken Test-Verteilung ab
- **Garantie:** Lineare Fehlerakkumulation (nicht quadratisch)

**Nachteil:** Experte muss online labelen (aufwendig, nicht immer möglich).

---

### ❓ TYPISCHE KLAUSURFRAGEN (Kompakt)

#### **Transformers**
1. **Q:** Self-Attention Q,K,V? **A:** Query sucht relevante Keys, Value enthält Info. Attention gewichtet Values nach Relevanz.
2. **Q:** Warum Positional Encoding? **A:** Attention ist positionsinvariant. PE addiert Positionsinfo (sinus/cosinus oder RoPE).
3. **Q:** Encoder vs Decoder? **A:** Encoder: bidirektional, Self-Attention. Decoder: kausal (masked), Self-Attention + Cross-Attention.
4. **Q:** Multi-Head? **A:** Parallele Heads, verschiedene Aspekte, concat + Output-Projektion.
5. **Q:** Masked Attention? **A:** Causal Mask (minus unendlich für Zukunft), verhindert Cheating bei Generation.
6. **Q:** Warum Transformer > RNN? **A:** O(1) Pfadlänge, Parallelisierung, kein Vanishing Gradient.
7. **Q:** SwiGLU vs GELU? **A:** SwiGLU: gated (Swish), 2 Projektionen, empirisch besser.

#### **LSTM & RNNs**
8. **Q:** 3 LSTM Gates? **A:** Forget (löschen), Input (hinzufügen), Output (filtern). Cell State ist gewichtete Summe aus altem Cell State und neuer Information.
9. **Q:** Vanishing Gradient? **A:** Gradient proportional zum Produkt der tanh-Ableitungen geht gegen null. LSTM: additiver Cell State löst.
10. **Q:** LSTM vs GRU? **A:** GRU: 2 Gates (Update, Reset), kein Cell State, weniger Parameter.
11. **Q:** Bidirectional LSTM? **A:** Forward + Backward, concat. Für Sequence Labeling (NER, POS).

#### **Word Embeddings**
12. **Q:** CBOW vs Skip-gram? **A:** CBOW: Kontext→Ziel (schnell). Skip-gram: Ziel→Kontext (besser für selten).
13. **Q:** Distributional Hypothesis? **A:** "Wort bekannt durch Kontext" (Firth 1957). Ähnliche Kontexte → ähnliche Vektoren.
14. **Q:** FastText vs Word2Vec? **A:** FastText: Subword n-grams, OOV-kompatibel, Morphologie.
15. **Q:** TF-IDF? **A:** tf ist lokal, idf ist global (Seltenheit). TF-IDF ist tf mal idf.
16. **Q:** BPE? **A:** Iterative Fusion häufigster Bigramme, kontrollierte Vokabulargröße.

#### **RL**
17. **Q:** Q-Learning Update? **A:** Q-Wert wird aktualisiert in Richtung von Reward plus diskontiertem maximalem nächsten Q-Wert. Off-Policy.
18. **Q:** Q-Learning vs SARSA? **A:** Q: maximaler nächster Q-Wert (off-policy, optimistisch). SARSA: tatsächlicher nächster Q-Wert (on-policy, vorsichtig).
19. **Q:** Overestimation DQN? **A:** max überschätzt. Double DQN: Selektion mit Q, Bewertung mit Target.
20. **Q:** Target Networks? **A:** Stabiler TD-Target (Gewichte periodisch kopiert oder Polyak).
21. **Q:** UCB? **A:** Mean-Reward + Bonus für Unsicherheit. Optimismus bei Unsicherheit.
22. **Q:** Thompson Sampling? **A:** Sample Posterior, wähle optimal unter Sample.
23. **Q:** Bellman? **A:** Wert ist Erwartung von Reward plus Diskont mal Wert des nächsten Zustands. Rekursiv.

#### **Generative Modelle**
24. **Q:** Mode Collapse GAN? **A:** Generator produziert 1 Mode. WGAN + Gradient Penalty löst.
25. **Q:** VAE Reparametrisierung? **A:** z ist Mittelwert plus Varianz mal Noise. Differenzierbar.
26. **Q:** Diffusion Forward/Reverse? **A:** Forward: Noise. Reverse: Netzwerk lernt Noise vorherzusagen.
27. **Q:** Classifier-Free Guidance? **A:** Linearkombination unconditional + conditional. Kein separater Classifier.

#### **XAI**
28. **Q:** LIME vs SHAP? **A:** LIME: lokal linear (Surrogate). SHAP: Shapley Values (fair, additiv).
29. **Q:** SHAP? **A:** Summe marginaler Beiträge über alle Subsets, fair gewichtet.
30. **Q:** PFI bei Korrelation? **A:** Korrelierte Features teilen Wichtigkeit. Lösung: Clustern.

#### **Imitation Learning**
31. **Q:** Distributional Shift BC? **A:** Training ungleich Test-Verteilung. Fehler quadratisch.
32. **Q:** DAgger? **A:** Rollout Policy, Experte labelt, Daten aggregiert. Löst Shift (linear).

---

### 🎓 DOZENTEN-HINWEISE

| Thema | Hinweis |
|-------|---------|
| **Transformer** | Positional Encoding kritisch! Attention is not all you need. |
| **LSTM** | Peephole Connections und Coupled Gates sind bekannte Varianten. |
| **RL** | GLIE garantiert Konvergenz bei Q-Learning. |
| **GAN** | Discriminator und Generator sollten vergleichbar stark sein. |
| **XAI** | Intrinsisch interpretierbare Modelle oft ausreichend. |
| **IL** | Causal Confusion: Policy lernt falsche Kausalitäten. |
| **Embeddings** | Bias in Embeddings verstärkt gesellschaftliche Vorurteile. |

---

*Erstellt: 17.03.2026 | Aktualisiert: 17.03.2026 13:24 | Basierend auf 8 PDF-Vorlesungsunterlagen*

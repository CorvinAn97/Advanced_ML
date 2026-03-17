# ZUSAMMENFASSUNG 01: Wiederholung - Grundlagen ML (Erweitert)

## Übersicht
- **Seitenzahl:** ~28 Seiten (AdvancedML-01-Wiederholung.pdf)
- **Hauptthemen:** Mathematische Grundlagen, Supervised Learning, Neuronale Netze, Backpropagation, Regularisierung
- **Prüfungsrelevanz:** 🟢 GRUNDWISSEN - Basis für alle weiteren Themen

---

## 1. Mathematische Grundlagen ⭐⭐⭐

### 1.1 Vektoren und Matrizen

#### Matrix-Notation
```
A ∈ ℝ^(m×n)  mit  A = [A_ij]
```
- **i:** Zeilenindex (1 bis m)
- **j:** Spaltenindex (1 bis n)
- **Element:** A_ij bezeichnet Element in Zeile i, Spalte j

**Beispiel:**
```
A = [A₁₁  A₁₂  A₁₃]    ∈ ℝ^(2×3)
    [A₂₁  A₂₂  A₂₃]

M = [M₁₁  M₁₂]         ∈ ℝ^(3×2)
    [M₂₁  M₂₂]
    [M₃₁  M₃₂]
```

#### Vektoren
```
x = [x₁  x₂  x₃]  ∈ ℝ^(1×3)   (Zeilenvektor)

y = [y₁]
    [y₂]           ∈ ℝ^(3×1)   (Spaltenvektor)
    [y₃]
```

---

### 1.2 Matrix-Multiplikation ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Grundregel: "Zeile auf Spalte"
```
C = A · M

C_ij = Σ_(k=1)^m A_ik · M_kj
```

**Voraussetzung:**
- #Spalten(A) = #Zeilen(M) = m
- A ∈ ℝ^(n×m), M ∈ ℝ^(m×p) → C ∈ ℝ^(n×p)

#### Beispielrechnung
```
A = [A₁₁  A₁₂  A₁₃]         M = [M₁₁  M₁₂]
    [A₂₁  A₂₂  A₂₃]             [M₂₁  M₂₂]
                              [M₃₁  M₃₂]

C = A · M = [A₁₁·M₁₁ + A₁₂·M₂₁ + A₁₃·M₃₁    A₁₁·M₁₂ + A₁₂·M₂₂ + A₁₃·M₃₂]
            [A₂₁·M₁₁ + A₂₂·M₂₁ + A₂₃·M₃₁    A₂₁·M₁₂ + A₂₂·M₂₂ + A₂₃·M₃₂]
```

#### Spezialfall: Matrix × Vektor
```
y = M · x  mit  M ∈ ℝ^(m×n), x ∈ ℝ^(n×1)

y₁ = M₁₁·x₁ + M₁₂·x₂ + ... + M₁n·x_n
y₂ = M₂₁·x₁ + M₂₂·x₂ + ... + M₂n·x_n
...
y_m = M_m₁·x₁ + M_m₂·x₂ + ... + M_mn·x_n

Resultat: y ∈ ℝ^(m×1) - gleiche Dimension wie Zeilen von M
```

#### Skalarprodukt als Matrixmultiplikation
```
θ · x = θ₁·x₁ + θ₂·x₂ + ... + θ_d·x_d

Mit θ, x ∈ ℝ^(d×1):
θ · x = θ^T · x  (Matrixmultiplikation 1×d · d×1 = 1×1 Skalar)
```

---

### 1.3 Partielle Ableitungen und Gradient ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Partielle Ableitung
Für Funktion f(θ₁, ..., θ_d) mit d Variablen:
```
∂f/∂θ_i  = Ableitung nach θ_i, alle anderen Variablen als Konstanten behandelt
```

**Beispiel:**
```
f(θ) = Σ_(i=1)^d θ_i · x_i = θ₁·x₁ + θ₂·x₂ + ... + θ_d·x_d

∂f/∂θ_i = x_i  (für i = 1, ..., d)
```

#### Gradient
```
∇_θ f(θ) = [∂f/∂θ₁]
           [∂f/∂θ₂]
           [  ...  ]
           [∂f/∂θ_d]  ∈ ℝ^(d×1)
```

**Eigenschaften:**
- ∇_θ f(θ) zeigt in Richtung der **stärksten Steigung** von f(θ)
- Bei Minimierung: Bewegen in **entgegengesetzte Richtung** (-∇_θ f(θ))
- **Kritische Punkte:** Lokale Minima/Maxima bei ∇_θ f(θ) = 0 (Nullvektor)

---

## 2. Supervised Learning Grundlagen ⭐⭐⭐

### 2.1 Grundkonzept

#### Trainingsdaten
```
D = {(x^(i), y^(i)); i = 1, ..., n}

n: Anzahl Trainingsdaten
d: Anzahl Features (Dimension von x)
```

**Notation:**
- **x^(i):** Input-Vector ("features") für Beispiel i
- **y^(i):** Target/Label für Beispiel i
- **X:** Raum der Input-Werte (z.B. ℝ^d)
- **Y:** Raum der Output-Werte (z.B. ℝ für Regression, {0,1} für Klassifikation)

#### Hypothese
```
h: X → Y

Ziel: Finde Funktion h, sodass h(x) eine "gute" Näherung für y liefert
```

**Beispiele:**
- **Regression:** x = Wohnfläche → y = Hauspreis (ℝ)
- **Klassifikation:** x = Tumorgröße → y = {benign, malign} ({0, 1})

---

### 2.2 Kostenfunktion (Least Squares) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Least-Squares Cost Function
```
J(θ) = (1/(2n)) · Σ_(i=1)^n (h_θ(x^(i)) - y^(i))²
```

**Komponenten:**
- **h_θ(x^(i)):** Vorhersage des Modells für Beispiel i
- **y^(i):** Tatsächlicher Wert (Ground Truth)
- **(h_θ(x^(i)) - y^(i))²:** Quadratischer Fehler pro Beispiel
- **1/(2n):** Normalisierung (Faktor 1/2 vereinfacht Ableitung)

#### Aufgabe
```
min_θ J(θ)

Wähle θ so, dass J(θ) minimal wird
→ Vorhersagefehler über alle Trainingsdaten minimiert
```

---

### 2.3 Gradient Descent ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Algorithmus
```
Starte mit initialem θ^(0) (beliebig)

Iterativ update:
θ_j ← θ_j - α · ∂J(θ)/∂θ_j   für j = 0, ..., d

Bis Konvergenz (J(θ) ändert sich kaum noch)
```

#### Parameter Update
```
∇_θ J(θ) = [∂J/∂θ₁]
           [∂J/∂θ₂]
           [  ...  ]
           [∂J/∂θ_d]

Update-Regel (Vektorform):
θ ← θ - α · ∇_θ J(θ)
```

**Intuition:**
- ∇_θ J(θ) zeigt Richtung des stärksten **Anstiegs** von J
- Subtraktion (-α·∇) bewegt θ in Richtung des stärksten **Abstiegs**
- α kontrolliert Schrittweite

---

### 2.4 Lernrate α ⭐ PRÜFUNGSRELEVANT

#### Wahl der Lernrate

**α zu klein:**
- Sehr langsame Konvergenz
- Viele Iterationen nötig
- Rechenintensiv

**α zu groß:**
- Divergenz möglich
- Oszillation um Minimum
- Überspringen des Optimums

**Visualisierung:**
```
J(θ)
  │
  │\       α zu groß: θ springt hin und her
  │ \      α optimal: konvergiert smoothly
  │  \     α zu klein: sehr langsame Fortschritte
  │   \___
  │       \
  └─────────────── θ
         θ*
```

#### Moderne Varianten
- **Adam:** Adaptive Lernraten pro Parameter
- **RMSprop:** Moving Average der Gradientenquadrate
- **Learning Rate Scheduling:** α wird während Training reduziert

---

## 3. Modell-Evaluation: Bias vs Variance ⭐⭐⭐ PRÜFUNGSRELEVANT

### 3.1 Grundkonzepte

#### Underfitting / High Bias
- **Modell zu einfach** für die Daten
- **Symptome:**
  - J_train groß (schlechte Performance auf Trainingsdaten)
  - J_CV groß (schlechte Performance auf Testdaten)
- **Ursache:** Modell kann Muster nicht erfassen

**Beispiel:** Lineare Regression für nicht-lineare Daten
```
h_θ(x) = θ₀ + θ₁·x  (zu einfach)
```

#### Overfitting / High Variance
- **Modell zu komplex** für die Daten
- **Symptome:**
  - J_train klein (sehr gute Performance auf Trainingsdaten)
  - J_CV groß (schlechte Performance auf Testdaten)
- **Ursache:** Modell lernt "Rauschen" statt Signal

**Beispiel:** Polynom zu hohen Grades
```
h_θ(x) = θ₀ + θ₁·x + θ₂·x² + ... + θ₁₀·x¹⁰  (zu komplex)
```

#### Optimal
- **J_train klein** (gute Trainings-Performance)
- **J_CV klein** (gute Test-Performance)
- **Generalisierung** gelingt

---

### 3.2 Diagnose über Fehler

| Szenario | J_train | J_CV | Diagnose | Lösung |
|----------|---------|------|----------|--------|
| Underfitting | groß | groß | High Bias | Komplexeres Modell, mehr Features |
| Overfitting | klein | groß | High Variance | Regularisierung, mehr Daten |
| Optimal | klein | klein | Good Fit | Weitermachen |

**Visualisierung:**
```
Underfitting:          Overfitting:           Optimal:
  x = train              x = train              x = train
  x = CV                 x = CV                 x = CV
     x                       x                      x
   x   x                   x   x                  x   x
      x                       x                      x
  ━━━ (linear)          ～～～ (wiggly)         〜〜～ (smooth)
```

---

### 3.3 Regularisierung ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Regularized Cost Function
```
J(θ) = (1/(2n)) · Σ_(i=1)^n (h_θ(x^(i)) - y^(i))² + λ · Σ_(j=1)^d θ_j²
```

**Komponenten:**
- **Erster Term:** Datenfit (Least Squares)
- **Zweiter Term:** Regularisierung (straft große Parameter)
- **λ (Lambda):** Regularisierungsparameter (Hyperparameter)

#### Einfluss von λ

**Großes λ:**
- Starke Strafe für große θ_j
- θ_j werden klein (nahe 0)
- **Risiko:** High Bias (Underfitting)
- Modell zu "glatt"

**Kleines λ:**
- Schwache Regularisierung
- θ_j können groß werden
- **Risiko:** High Variance (Overfitting)
- Modell zu "wiggly"

**Optimales λ:**
- Balance zwischen Bias und Variance
- Durch Cross-Validation finden

---

#### Visualisierung λ-Einfluss
```
J_CV
  │
  │\
  │ \
  │  \___  Optimum hier
  │      \
  │       \
  └─────────────── λ
  Klein       Groß
  (High       (High
  Variance)   Bias)
```

---

### 3.4 Weitere Regularisierungsmethoden

#### Dropout
- Zufälliges Deaktivieren von Neuronen während Training
- Reduziert Co-Adaptation von Features
- Besonders effektiv in Fully Connected Layers

#### Early Stopping
- Training stoppen wenn Validation Error steigt
- Verhindert Overfitting durch zu langes Training

#### Batch Normalization
- Normalisierung von Activations über Batch
- Reduziert Internal Covariate Shift
- Ermöglicht höhere Lernraten

#### Layer Normalization
- Normalisierung über Feature-Dimension (nicht Batch)
- Wichtig für RNNs/Transformers
- Unabhängig von Batch-Größe

---

## 4. Fully Connected Networks (MLP) ⭐⭐⭐

### 4.1 Künstliches Neuron ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Grundmodell
```
Inputs:    x = [x₁, x₂, ..., x_d]^T
Weights:   θ = [θ₁, θ₂, ..., θ_d]^T
Output:    h_θ(x) = g(θ^T · x)
```

**Komponenten:**
- **x:** Feature-Vektor (Inputs)
- **θ:** Lernbare Parameter (Weights)
- **g:** Aktivierungsfunktion (nicht-linear)

#### Bias-Term
```
x₀ = 1  (implizit immer dabei)
θ₀: Bias-Parameter

h_θ(x) = g(θ₀·1 + θ₁·x₁ + ... + θ_d·x_d)
       = g(θ^T · x)  mit x₀ = 1
```

---

### 4.2 Multilayer Perceptron (MLP) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Architektur
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Eigenschaften:**
- **Fully Connected:** Alle Neuronen einer Schicht mit allen der nächsten verbunden
- **Schichtweiser Aufbau:** Information fließt vorwärts (Feedforward)
- **Beliebig viele Hidden Layers** möglich (Deep Learning)

#### Visualisierung
```
Layer 1         Layer 2         Layer 3
"Input"         "Hidden"        "Output"
  ○  ────────────  ○  ────────────  ○
  ○  ────────────  ○  ────────────  ○
  ○  ────────────  ○
  ○
```

---

### 4.3 Notation für MLP ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Indizes
```
l: Layer-Index (l = 1, 2, ..., L)
i: Neuron-Index innerhalb von Layer l
```

#### Variablen
```
z_i^(l): Input der Aktivierung i in Layer l
a_i^(l): Output der Aktivierung i in Layer l
         a_i^(l) = g(z_i^(l))

Θ^(l): Parametermatrix von Layer l zu Layer l+1
       Θ^(l) ∈ ℝ^(q_(l+1) × (q_l + 1))
       (+1 wegen Bias-Term a₀^(l) = 1)
```

#### Forward Propagation
```
Für Layer l → l+1:

z_i^(l+1) = Σ_j Θ_ij^(l) · a_j^(l)
a_i^(l+1) = g(z_i^(l+1))

In Matrixform:
z^(l+1) = Θ^(l) · a^(l)
a^(l+1) = g(z^(l+1))
```

#### Dimensionsbetrachtung
```
q_l: Anzahl Einheiten in Layer l
q_(l+1): Anzahl Einheiten in Layer l+1

Θ^(l): q_(l+1) × (q_l + 1) Matrix

Beispiel:
Layer 1: 64 Units
Layer 2: 128 Units
Θ^(1): 128 × 65 Matrix (inkl. Bias)
Parameter: 128 × 65 = 8.320
```

---

## 5. Backpropagation ⭐⭐⭐ PRÜFUNGSRELEVANT

### 5.1 Kernidee

**Ziel:** Berechne Gradienten ∂J/∂Θ_ij^(l) für alle Parameter

**Methode:** Kettenregel rückwärts durch das Netzwerk

**Intuition:**
- Fehler am Output wird zurück durch das Netz propagiert
- Jedes Neuron erhält "Schuldzuweisung" (δ)
- Gradient = δ × Aktivierung des vorherigen Neurons

---

### 5.2 Backpropagation-Formeln ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Output Layer (L)
```
δ_i^(L) = a_i^(L) - y_i  (für Least Squares)
         = ∂J/∂z_i^(L)
```

#### Hidden Layers (l = L-1, ..., 2)
```
δ_j^(l) = Σ_i δ_i^(l+1) · Θ_ij^(l) · g'(z_j^(l))

In Matrixform:
δ^(l) = (Θ^(l))^T · δ^(l+1) ⊙ g'(z^(l))

⊙: Hadamard-Produkt (elementweise Multiplikation)
```

#### Gradienten
```
∂J/∂Θ_ji^(l) = δ_j^(l+1) · a_i^(l)

In Matrixform:
∇_(Θ^(l)) J = δ^(l+1) · (a^(l))^T
```

---

### 5.3 Backpropagation Algorithmus

```
1. Forward Pass:
   Für l = 1 bis L:
     z^(l) = Θ^(l-1) · a^(l-1)
     a^(l) = g(z^(l))

2. Backward Pass:
   δ^(L) = ∇_a J ⊙ g'(z^(L))  (Output Error)
   
   Für l = L-1 bis 2:
     δ^(l) = (Θ^(l))^T · δ^(l+1) ⊙ g'(z^(l))

3. Gradienten berechnen:
   ∇_(Θ^(l)) J = δ^(l+1) · (a^(l))^T
```

---

### 5.4 Anschauliche Darstellung

```
Forward:  x → [Layer 1] → a^(1) → [Layer 2] → a^(2) → ... → Output
                                                          ↓
Backward: x ← [Layer 1] ← δ^(1) ← [Layer 2] ← δ^(2) ← ... ← δ^(L)

δ^(L): Error am Output
δ^(l): "Wie viel Schuld trägt Layer l am Fehler?"
```

---

### 5.5 Komponentenweise Multiplikation (Hadamard)

```
a ⊙ b = [a₁·b₁]
        [a₂·b₂]
        [ ... ]
        [a_d·b_d]

In Backpropagation:
δ^(l) = ... ⊙ g'(z^(l))

g'(z^(l)): Ableitung der Aktivierung an Position z^(l)
```

---

## 6. Aktivierungsfunktionen ⭐⭐⭐ PRÜFUNGSRELEVANT

### 6.1 Sigmoid

#### Definition
```
sigmoid(x) = 1 / (1 + e^(-x))  ∈ [0, 1]
```

#### Eigenschaften
- **Wertebereich:** [0, 1] (Wahrscheinlichkeits-Interpretation)
- **Nullpunkt:** sigmoid(0) = 0.5
- **Symmetrie:** Punktsymmetrisch um (0, 0.5)

#### Ableitung
```
g'(x) = sigmoid(x) · (1 - sigmoid(x))
      ∈ (0, 0.25]
```

#### Problem: Sättigung
```
Für |x| >> 0:
- sigmoid(x) ≈ 0 oder 1
- g'(x) ≈ 0 (Gradient verschwindet!)
- Training sehr langsam oder stagniert
```

**Visualisierung:**
```
g(x)
 1│     ━━━━━━━━━
  │    /
  │   /
0.5│  /
  │ /
  │/
 0└────────────── x
  -5     0     5

g'(x)
   │
0.25│   /\
    │  /  \
    │ /    \
   0└──────── x
     -5  0  5
```

---

### 6.2 tanh (Tangens hyperbolicus)

#### Definition
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))  ∈ [-1, 1]
```

#### Eigenschaften
- **Wertebereich:** [-1, 1]
- **Null-zentriert:** tanh(0) = 0
- **Vorteil gegenüber Sigmoid:** Gradient stärker (bis 1)

#### Ableitung
```
g'(x) = 1 - tanh²(x)
      ∈ (0, 1]
```

#### Problem: Sättigung
```
Für |x| >> 0:
- tanh(x) ≈ -1 oder 1
- g'(x) ≈ 0 (Gradient verschwindet!)
```

**Einsatz:** Nützlich für RNNs (begrenzter Wertebereich)

---

### 6.3 ReLU (Rectified Linear Unit) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Definition
```
ReLU(x) = max(0, x)

g'(x) = 1  für x > 0
      = 0  für x < 0
      = ?  für x = 0 (meist 0 oder 1)
```

#### Eigenschaften
- **Wertebereich:** [0, ∞)
- **Nicht linear:** Aber stückweise linear
- **Kein Sättigungsproblem** für x > 0

#### Vorteile
- **Effizient:** Einfache Berechnung
- **Kein Vanishing Gradient** für positive Werte
- **Sparse Activations:** Viele Neuronen inaktiv (0)

#### Problem: "Tode Neuronen" (Dying ReLU)
```
Für x < 0:
- ReLU(x) = 0
- g'(x) = 0
- Gradient verschwindet komplett
- Neuron lernt nicht mehr ("tot")
```

**Lösungen:**
- **Leaky ReLU:** g(x) = max(α·x, x) mit α klein (z.B. 0.01)
- **Parametric ReLU:** α lernbar

---

### 6.4 Softmax (Mehrklassenklassifizierung) ⭐⭐⭐ PRÜFUNGSRELEVANT

#### Definition
```
softmax(z)_i = e^(z_i) / Σ_(j=1)^k e^(z_j)

Eigenschaften:
- Σ_i softmax(z)_i = 1 (Wahrscheinlichkeitsverteilung)
- softmax(z)_i ∈ (0, 1)
```

#### Beispiel
```
z = [1]
    [2]
    [4]

softmax(z) = [e¹ / (e¹ + e² + e⁴)]   [0.042]
             [e² / (e¹ + e² + e⁴)] = [0.114]
             [e⁴ / (e¹ + e² + e⁴)]   [0.844]
```

#### Einsatz
- **Output Layer** für Mehrklassenklassifizierung
- **Cross-Entropy Loss** als Kostenfunktion
- Interpretation als Klassenwahrscheinlichkeiten

---

## 7. Zusammenfassung & Prüfungsrelevanz

### 7.1 Wichtigste Konzepte ⭐⭐⭐

| Konzept | Prüfungsrelevanz | Typische Frage |
|---------|------------------|----------------|
| **Matrix-Multiplikation** | ⭐⭐⭐ | "Zeile auf Spalte", Dimensionsbedingung |
| **Gradient** | ⭐⭐⭐ | Richtung stärkster Steigung |
| **Gradient Descent** | ⭐⭐⭐ | Update-Regel, Lernrate α |
| **Bias vs Variance** | ⭐⭐⭐ | Underfitting vs Overfitting Diagnose |
| **Regularisierung** | ⭐⭐⭐ | Einfluss von λ |
| **MLP Architektur** | ⭐⭐⭐ | Fully Connected, Schichten |
| **Backpropagation** | ⭐⭐⭐ | Kettenregel, δ Berechnung |
| **Aktivierungsfunktionen** | ⭐⭐⭐ | Sigmoid, tanh, ReLU, Softmax |

---

### 7.2 Wichtige Formeln ⭐⭐⭐

#### Matrix-Multiplikation
```
C_ij = Σ_(k=1)^m A_ik · M_kj
```

#### Gradient
```
∇_θ f(θ) = [∂f/∂θ₁, ∂f/∂θ₂, ..., ∂f/∂θ_d]^T
```

#### Gradient Descent
```
θ_j ← θ_j - α · ∂J(θ)/∂θ_j
```

#### Least Squares Cost Function
```
J(θ) = (1/(2n)) · Σ_(i=1)^n (h_θ(x^(i)) - y^(i))²
```

#### Regularized Cost Function
```
J(θ) = (1/(2n)) · Σ_(i=1)^n (h_θ(x^(i)) - y^(i))² + λ · Σ_(j=1)^d θ_j²
```

#### Backpropagation
```
δ_j^(l) = Σ_i δ_i^(l+1) · Θ_ij^(l) · g'(z_j^(l))
∂J/∂Θ_ji^(l) = δ_j^(l+1) · a_i^(l)
```

#### Sigmoid
```
sigmoid(x) = 1 / (1 + e^(-x))
sigmoid'(x) = sigmoid(x) · (1 - sigmoid(x))
```

#### tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

#### ReLU
```
ReLU(x) = max(0, x)
```

#### Softmax
```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```

---

### 7.3 Typische Prüfungsfragen

**Frage 1: Gradient**
> "Was zeigt der Gradient ∇_θ f(θ) an?"

**Antwort:**
- Richtung der stärksten Steigung von f(θ)
- Bei Minimierung: Bewegen in entgegengesetzte Richtung

**Frage 2: Gradient Descent**
> "Wie funktioniert Gradient Descent?"

**Antwort:**
- Starte mit initialem θ
- Iterativ: θ ← θ - α · ∇_θ J(θ)
- Bis Konvergenz

**Frage 3: Bias vs Variance**
> "Wie diagnostiziert man High Bias vs High Variance?"

**Antwort:**
- High Bias: J_train groß, J_CV groß → Underfitting
- High Variance: J_train klein, J_CV groß → Overfitting

**Frage 4: Regularisierung**
> "Wie wirkt sich λ auf das Modell aus?"

**Antwort:**
- Großes λ: Hoher Bias (stark regularisiert)
- Kleines λ: Hohe Variance (schwach regularisiert)

**Frage 5: Backpropagation**
> "Wie funktioniert Backpropagation?"

**Antwort:**
- Fehler δ am Output berechnen
- Rückwärts durch Netz propagieren: δ^(l) = (Θ^(l))^T · δ^(l+1) ⊙ g'(z^(l))
- Gradient: ∂J/∂Θ = δ · a^T

**Frage 6: Aktivierungsfunktionen**
> "Warum haben Sigmoid/tanh Sättigungsprobleme?"

**Antwort:**
- Für |x| >> 0: Output ≈ konstant
- Ableitung g'(x) ≈ 0
- Gradient verschwindet → Training stagniert

**Frage 7: MLP Notation**
> "Was ist a_i^(l) und Θ^(l)?"

**Antwort:**
- a_i^(l): Output von Neuron i in Layer l
- Θ^(l): Parametermatrix von Layer l zu l+1

---

## 8. Eigene Notizen & Verständnis

### 8.1 Kernpunkte

✅ **Gradient:** Zeigt Richtung stärkster Steigung (für Minimierung: entgegen)
✅ **Gradient Descent:** Iterative Optimierung mit Lernrate α
✅ **Bias-Variance:** Tradeoff zwischen Underfitting und Overfitting
✅ **Regularisierung:** λ kontrolliert Modellkomplexität
✅ **Backpropagation:** Kettenregel rückwärts durch Netz
✅ **Aktivierungen:** Nicht-Linearität für expressive Modelle

### 8.2 Häufige Fehler

❌ Lernrate zu groß → Divergenz
❌ Lernrate zu klein → Sehr langsam
❌ Keine Regularisierung bei komplexen Modellen → Overfitting
❌ Backpropagation-Fehler: δ falsch berechnet
❌ Aktivierung ohne Ableitung in Backprop

### 8.3 Lernstrategie

1. **Formeln auswendig:** Gradient Descent, Regularization, Backprop
2. **Konzepte verstehen:** Warum Regularisierung? Warum Backprop?
3. **Diagnose können:** Bias vs Variance identifizieren
4. **Ableitungen kennen:** Sigmoid, tanh, ReLU
5. **Dimensionsbetrachtung:** Θ-Matrizen korrekt dimensionieren

---

**Erstellt:** 2026-03-17 (erweiterte Version)
**Basierend auf:** AdvancedML-01-Wiederholung.pdf (~28 Seiten)
**Umfang:** Vollständige Abdeckung aller PDF-Themen

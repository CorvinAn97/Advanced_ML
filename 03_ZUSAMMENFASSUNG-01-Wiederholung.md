# ZUSAMMENFASSUNG 01: Wiederholung - Grundlagen ML

## Übersicht
- Seitenzahl: ~28 Seiten
- Hauptthemen: Mathematische Grundlagen, Supervised Learning, Neuronale Netze, Backpropagation, Regularisierung

## Detaillierte Inhalte

### 1. Mathematische Grundlagen

#### Vektoren und Matrizen
- **Matrix-Notation:** A ∈ ℝ^(m×n), Elemente A_ij (Zeile i, Spalte j)
- **Matrix-Vektor-Multiplikation:** "Zeile auf Spalte" Prinzip
- **Dimensionen:** Für A·M muss #Spalten(A) = #Zeilen(M) sein
- **Spezialfall Matrix x Vektor:** Resultat ist Vektor der gleichen Dimension wie Zeilen der Matrix

#### Partielle Ableitungen und Gradient
- **Partielle Ableitung:** ∂f/∂θ_i - Ableitung nach einer Variable, andere konstant
- **Gradient:** ∇_θ f(θ) = [∂f/∂θ_1, ..., ∂f/∂θ_d]^T
- **Eigenschaft:** Gradient zeigt Richtung der stärksten Steigung
- **Kritische Punkte:** Lokale Minima/Maxima bei ∇_θ f(θ) = 0

### 2. Supervised Learning Grundlagen

#### Grundkonzept
- **Trainingsdaten:** {(x^(i), y^(i)); i = 1, ..., n}
- **Ziel:** Funktion h: X → Y finden, die "gute" Näherung liefert
- **h wird Hypothese genannt**

#### Kostenfunktion (Least Squares)
```
J(θ) = (1/2n) Σ_{i=1}^n (h_θ(x^(i)) - y^(i))^2
```
- Misst Abweichung zwischen Vorhersage und tatsächlichem Wert
- Aufgabe: θ so wählen, dass J(θ) minimiert wird

#### Gradient Descent
```
θ_j ← θ_j - α * ∂J(θ)/∂θ_j
```
- **Lernrate α:** Schrittweite der Optimierung
- **Problem α zu klein:** Sehr langsame Konvergenz
- **Problem α zu groß:** Divergenz möglich

### 3. Modell-Evaluation: Bias vs Variance

#### Konzepte
- **Underfitting / High Bias:** Modell zu einfach, schlechte Performance auf Trainings- und Testdaten
- **Overfitting / High Variance:** Modell zu komplex, gut auf Trainingsdaten, schlecht auf Testdaten

#### Diagnose über Fehler
- **J_train groß, J_CV groß:** High Bias (Underfitting)
- **J_train klein, J_CV groß:** High Variance (Overfitting)
- **J_train klein, J_CV klein:** Optimal

#### Regularisierung
```
J(θ) = (1/2n) Σ (h_θ(x^(i)) - y^(i))^2 + λ Σ_{j=1}^d θ_j^2
```
- **λ (Lambda):** Regularisierungsparameter
- **Großes λ:** Hoher Bias (starkes Strafen großer Parameter)
- **Kleines λ:** Hohe Varianz (schwache Regularisierung)

#### Weitere Regularisierungsmethoden
- Dropout
- Early Stopping
- Batch Normalization
- Layer Normalization

### 4. Fully Connected Networks (MLP)

#### Künstliches Neuron
```
h_θ(x) = g(θ^T x)
```
- **Inputs x:** Feature-Vektor
- **Weights θ:** Lernbare Parameter
- **g:** Aktivierungsfunktion (z.B. Sigmoid, ReLU, tanh)

#### Multilayer Perceptron (MLP)
- **Struktur:** Input Layer → Hidden Layer(s) → Output Layer
- **Fully Connected:** Alle Neuronen einer Schicht mit allen der nächsten verbunden
- **Aktivierungen a_i^(l):** Output des Neurons i in Layer l
- **Bias-Term:** x_0 = 1 implizit in jedem Layer

#### Notation
- **z_i^(l):** Input der Aktivierung i in Layer l
- **a_i^(l):** Output der Aktivierung i in Layer l (a_i^(l) = g(z_i^(l)))
- **Θ^(l):** Parametervektor/ Matrix für Layer l zu l+1

### 5. Backpropagation

#### Kernidee
- Berechnung der Gradienten über Kettenregel
- Fehler wird vom Output zurück durch das Netz propagiert

#### Backpropagation-Formel
```
δ_j^(l) = Σ_{i} δ_i^(l+1) * Θ_ij^(l) * g'(z_j^(l))
∂J/∂Θ_ji^(l) = δ_j^(l+1) * a_i^(l)
```

#### Komponentenweise Multiplikation
- a ⊙ b: Elementweise Multiplikation (Hadamard-Produkt)

### 6. Aktivierungsfunktionen

#### Sigmoid
```
sigmoid(x) = 1 / (1 + e^(-x)) ∈ [0, 1]
```
- Problem: Sättigung für |x| >> 0 → Gradient ≈ 0

#### tanh (Tangens hyperbolicus)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) ∈ [-1, 1]
```
- Null-zentriert, aber auch Sättigungsproblem

#### ReLU
```
ReLU(x) = max(0, x)
```
- Kein Sättigungsproblem für x > 0
- Problem: "Tote Neuronen" für x < 0

#### Softmax (für Mehrklassenklassifizierung)
```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```
- Wahrscheinlichkeitsinterpretation: Σ_i softmax(z)_i = 1

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ Gradient Descent
- Warum: Kernalgorithmus des trainings
- Was: Update-Regel, Lernrate, Konvergenzverhalten

### ✅ Backpropagation
- Warum: Fundamentales Verständnis für NN-Training nötig
- Was: Kettenregel, Fehler-Rückwärtspropagation

### ✅ Bias vs Variance Tradeoff
- Warum: Entscheidend für Modell-Selection
- Was: Diagnosemethoden, Regularisierung

### ✅ Aktivierungsfunktionen
- Warum: Einfluss auf Lernverhalten
- Was: Vor- und Nachteile verschiedener Funktionen

## Formeln/Algorithmen (wichtig)

### Kostenfunktion (Least Squares)
```
J(θ) = (1/2n) Σ_{i=1}^n (h_θ(x^(i)) - y^(i))^2
```

### Gradient Descent Update
```
θ_j ← θ_j - α * ∂J(θ)/∂θ_j
```

### Regularisierte Kostenfunktion
```
J(θ) = (1/2n) Σ (h_θ(x^(i)) - y^(i))^2 + λ Σ_{j=1}^d θ_j^2
```

### Softmax
```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **Gradient Descent** ist die Arbeitspferd-Optimierung in ML
- **Backpropagation** = Kettenregel angewendet auf neuronale Netze
- **Bias-Variance-Tradeoff** bestimmt Modellkomplexität
- **Regularisierung** verhindert Overfitting durch Bestrafung großer Parameter

### ⚠️ Häufige Fehler
- Lernrate zu groß → Divergenz
- Lernrate zu klein → Sehr langsames Training
- Keine Regularisierung bei komplexen Modellen → Overfitting

### 📝 Prüfungsrelevante Fragen
1. Wie funktioniert Gradient Descent?
2. Was ist der Unterschied zwischen High Bias und High Variance?
3. Wie wirkt sich λ auf das Modell aus?
4. Warum haben Sigmoid/tanh Sättigungsprobleme?
5. Wie funktioniert Backpropagation?

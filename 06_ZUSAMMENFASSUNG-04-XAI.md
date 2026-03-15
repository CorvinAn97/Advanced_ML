# ZUSAMMENFASSUNG 04: Explainable AI (XAI)

## Übersicht
- Seitenzahl: ~55 Seiten
- Hauptthemen: Interpretierbare Modelle, LIME, SHAP, Counterfactuals, LRP, Integrated Gradients

## Detaillierte Inhalte

### 1. Einführung in XAI

#### Motivation
- ML-Modelle werden immer komplexer (Deep Learning)
- Einsatz in sicherheitskritischen Bereichen (autonomes Fahren, Gesundheitswesen)
- "Black Boxes" schwer zu verstehen
- Wichtig für Entwickler UND Endnutzer

#### Interpretable ML vs Explainable AI
- **Interpretable ML:** Modellvorhersagen für Entwickler verständlich machen
- **Explainable AI:** Vorhersagen allgemeinverständlich erklären (auch für Nicht-ML-Experten)

#### Anforderungen an gute Erklärungen
1. **Vollständigkeit:** Alle Modelloutputs erklärbar
2. **Genauigkeit:** Gründe reflektieren Modellentscheidung
3. **Sinnhaftigkeit:** Stakeholder können Erklärung verstehen
4. **Konsistenz:** Reproduzierbar, stabil gegen kleine Input-Änderungen

---

### 2. Taxonomie von Interpretierbarkeit

#### Intrinsisch vs Post-hoc
- **Intrinsisch:** Modell ist von Natur aus interpretierbar
- **Post-hoc:** Erklärung wird nachträglich erstellt

#### Lokal vs Global
- **Lokal:** Einzelne Vorhersagen erklären
- **Global:** Gesamtverhalten des Modells verstehen

#### Modellspezifisch vs Modellunabhängig
- **Modellspezifisch:** Nutzt Modell-Interna
- **Modellunabhängig:** Funktioniert mit beliebigen Modellen

---

### 3. Intrinsisch interpretierbare Modelle

#### Lineare Regression
```
y = w₀ + w₁x₁ + ... + w_dx_d
```
- Parameter w_i zeigen direkt Einfluss auf Output
- Problem bei korrelierten Inputs

#### Logistische Regression
```
p = sigmoid(z) = 1 / (1 + e^(-z))
z = w₀ + w₁x₁ + ... + w_dx_d
```
- Entscheidungsgrenze linear
- Einfluss auf Odds: exp(w_i)

#### Entscheidungsbäume
- Leicht interpretierbar durch Verfolgung der Knoten
- Voraussetzung: Baum nicht zu tief

#### Feature Importance bei Bäumen
```
Importance(f) = Σ (n_node / n_total) × ΔImpurity
```
- Summe aller Verbesserungen durch Splits mit diesem Feature
- Bei Ensembles: Summe über alle Bäume

**Probleme:**
- Verzerrung durch viele mögliche Splits
- Abhängigkeit von Baumtiefe
- Nur globale Werte, keine Richtung

---

### 4. Permutation Feature Importance (PFI)

#### Idee
- Feature ist wichtig, wenn Modell ohne dieses Feature deutlich schlechter vorhersagt

#### Vorgehen
1. **Baseline:** Miss Modellgüte auf Testdaten
2. **Permutiere:** Mische Werte eines Features zufällig durch
3. **Vergleiche:** Berechne Modellgüte erneut
4. **Wichtigkeit** = Abfall der Güte

#### Eigenschaften
- Funktioniert für jedes Modell
- Misst Einfluss auf tatsächliche Vorhersageleistung
- Rechenaufwändig (mehrere Auswertungen nötig)

#### Probleme
- Stark korrelierte Features: Wichtigkeit wird geteilt
- Zufallsabhängigkeit (mehrfach ausführen und mitteln)
- Muss auf Testdaten berechnet werden

---

### 5. LIME (Local Interpretable Model-agnostic Explanations)

#### Ziel
- Einzelne Vorhersagen beliebiger Modelle erklären
- Modellunabhängig

#### Idee
- Komplexe Entscheidungsgrenze global
- **Lokal** einfach zu approximieren (z.B. linear)

#### Vorgehen
1. **Störe** den zu erklärenden Datenpunkt (kleine zufällige Manipulationen)
2. **Generiere** synthetischen Datensatz
3. **Label** synthetische Daten mit komplexem Modell
4. **Trainiere** einfaches, interpretierbares Modell (Surrogate)
   - Gewichtung: Je näher am Original, desto höheres Gewicht

#### Mathematisch
```
g = argmin_{g∈G} L(f, g, π_x) + Ω(g)
```
- **L:** Güte der Näherung (z.B. MSE)
- **π_x:** Nähe zur Erklärung (z.B. exponentiell abnehmend)
- **Ω(g):** Komplexitätsmaß (z.B. L1-Regularisierung)

#### Eigenschaften
- Modellunabhängig
- Lokale Erklärungen
- Kann auf verschiedene Datentypen angewendet werden (Tabellarisch, Text, Bilder)

#### Beschränkungen
- Rechenintensiv
- Empfindlich gegen Perturbationseinstellungen
- Lokale Treue ≠ globales Verhalten

---

### 6. SHAP (SHapley Additive exPlanations)

#### Grundidee
- Basierend auf Shapley-Werten aus kooperativer Spieltheorie
- Jedes Feature = Spieler
- Fairer Beitrag jedes Features zur Vorhersage

#### Shapley Values Berechnung
```
φ_i = Σ_{S⊆N\{i}} (|S|!(|N|-|S|-1)! / |N|!) × [v(S∪{i}) - v(S)]
```
- **N:** Menge aller Features
- **S:** Teilmenge ohne Feature i
- **v(S):** Modellvorhersage mit Features aus S
- **v(S∪{i}):** Modellvorhersage mit S und Feature i

#### Marginaler Beitrag
- Beitrag von Feature i zu jeder möglichen Koalition S
- Mittelung über alle Teilmengen

#### Fehlende Features
- Für fehlende Features: Werte aus Hintergrund-Datensatz zufällig wählen

#### Optimierungen
- **Sampling-basiert:** Nur zufällige Teilmengen
- **KernelSHAP:** Modellunabhängig
- **TreeSHAP:** Für Baum-basierte Modelle (exakt, effizient)
- **DeepSHAP:** Für neuronale Netze

#### Eigenschaften
- **Additiv:** Summe aller SHAP-Werte + Baseline = Vorhersage
- **Konsistent:** Erklärungen sind stabil
- **Fair:** Berücksichtigt alle möglichen Feature-Kombinationen

#### Plot-Typen
- **Summary Plot (Beeswarm):** SHAP-Werte aller Samples, sortiert nach Wichtigkeit
- **Bar Plot:** Durchschnittliche absolute SHAP-Werte
- **Dependence Plot:** Einzelne Features vs SHAP-Werte
- **Waterfall Plot:** Einzelne Erklärung für einen Sample
- **Force Plot:** Interaktive Darstellung der Beiträge

---

### 7. Counterfactual Explanations

#### Idee
- "Was müsste sich ändern, damit die Vorhersage anders wäre?"
- Minimale Änderung am Input für gewünschte Output-Änderung

#### Beispiel
- Aktuell: "Kredit abgelehnt"
- Counterfactual: "Wenn das Einkommen 5% höher wäre → Kredit genehmigt"

#### Anforderungen
- **Proximity:** Änderung soll minimal sein
- **Plausibilität:** Counterfactual soll realistisch sein
- **Sparsity:** Möglichst wenige Features ändern

---

### 8. Layer-wise Relevance Propagation (LRP)

#### Idee
- Relevanz vom Output zurück durch das Netzwerk propagieren
- Jedes Neuron erhält Relevanz-Score

#### Regeln
- **LRP-0:** Gleichmäßige Verteilung
- **LRP-ε:** Stabilisierung durch ε
- **LRP-γ:** Bevorzugung positiver Beiträge

#### Anwendung
- Bilder: Heatmaps zeigen relevante Pixel
- Text: Wichtige Wörter hervorheben

---

### 9. Integrated Gradients (IG)

#### Idee
- Gradienten entlang eines Pfads von Baseline zu Input integrieren

#### Formel
```
IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα
```
- **x:** Input
- **x':** Baseline (z.B. schwarzes Bild)
- **α:** Interpolationsparameter

#### Eigenschaften
- **Sättigung:** Leere Baseline → IG = 0
- **Additivität:** Summe der IG-Werte = F(x) - F(x')

#### Axiome
- Sensitivity (wenn Input und Baseline unterschiedlich → Erklärung ≠ 0)
- Implementation Invariance
- Completeness

---

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ LIME
- Warum: Wichtigste Post-hoc Erklärungsmethode
- Was: Lokale Approximation, Surrogate-Modell, Perturbation

### ✅ SHAP
- Warum: Theoretisch fundiert, konsistent
- Was: Shapley-Werte, Berechnung, Plot-Typen

### ✅ Permutation Feature Importance
- Warum: Modellunabhängig, intuitiv
- Was: Baseline, Permutation, Abfall der Güte

### ✅ Intrinsisch interpretierbare Modelle
- Warum: Grundlagen
- Was: Lineare/logistische Regression, Entscheidungsbäume

## Formeln/Algorithmen (wichtig)

### Lineare Regression
```
y = w₀ + w₁x₁ + ... + w_dx_d
```

### Logistische Regression
```
p = 1 / (1 + e^(-(w₀ + w₁x₁ + ... + w_dx_d)))
```

### LIME Optimierung
```
g = argmin_{g∈G} L(f, g, π_x) + Ω(g)
```

### Shapley Value
```
φ_i = Σ_{S⊆N\{i}} (|S|!(|N|-|S|-1)! / |N|!) × [v(S∪{i}) - v(S)]
```

### Integrated Gradients
```
IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **LIME:** Lokale Approximation durch einfaches Surrogate-Modell
- **SHAP:** Fairer Beitrag basierend auf kooperativer Spieltheorie
- **PFI:** Einfach, aber rechenaufwändig
- **Counterfactuals:** "Was wäre wenn?" Erklärungen

### ⚠️ Häufige Fehler
- LIME-Parameter (Nähe, Komplexität) falsch wählen
- SHAP mit ungeeignetem Hintergrund-Datensatz
- PFI auf Trainingsdaten berechnen (Overfitting!)

### 📝 Prüfungsrelevante Fragen
1. Was ist der Unterschied zwischen intrinsisch und post-hoc interpretierbar?
2. Wie funktioniert LIME?
3. Was sind Shapley-Werte und wie werden sie berechnet?
4. Was ist Permutation Feature Importance?
5. Was sind Counterfactual Explanations?
6. Was sind die Axiome von Integrated Gradients?

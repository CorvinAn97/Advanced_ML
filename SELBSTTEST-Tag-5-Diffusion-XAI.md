# Selbsttest Tag 5: Diffusionsmodelle & Explainable AI (XAI)

**Umfang:** Ausführlicher Test zu allen Themen des Tages
**Zeitansatz:** 50-70 Minuten
**Hinweis:** Antworten nicht einfach ablesen - erst selbst überlegen, dann nachschlagen!

---

## Teil A: Grundkonzepte & Verständnis (15 Fragen)

### 1. Diffusionsmodelle - Grundlagen

**Frage 1:**
Erklären Sie das Grundkonzept von Diffusionsmodellen. Was sind die beiden Hauptprozesse und wie funktionieren sie zusammen?

**Frage 2:**
Was ist der Forward Process (Diffusion Process) bei Diffusionsmodellen? Beschreiben Sie, wie Rauschen schrittweise hinzugefügt wird.

**Frage 3:**
Warum können wir im Forward Process direkt von x₀ zu x_t springen, ohne alle Zwischenschritte zu berechnen? Welche Formel ermöglicht das?

**Frage 4:**
Was passiert im Reverse Process (Denoising)? Was muss das Modell lernen und wie wird es trainiert?

**Frage 5:**
Warum wird beim Training des Diffusionsmodells nicht direkt x₀ vorhergesagt, sondern das Rauschen ε? Was wäre der Nachteil einer direkten x₀-Vorhersage?

---

### 2. DDPM, DDIM und Sampling

**Frage 6:**
Was bedeutet DDPM und wie funktioniert der DDPM Sampling-Algorithmus? Erklären Sie die einzelnen Schritte.

**Frage 7:**
Warum ist DDPM-Sampling langsam? Wie viele Schritte sind typischerweise nötig und warum führt man nicht einfach weniger Schritte durch?

**Frage 8:**
Was ist DDIM und wie unterscheidet es sich von DDPM? Welchen entscheidenden Vorteil bietet DDIM?

**Frage 9:**
Warum ermöglicht DDIM das Überspringen von Schritten (z.B. nur 50 statt 1000), während DDPM das nicht kann?

**Frage 10:**
Erklären Sie den Unterschied zwischen stochastischem und deterministischem Sampling bei Diffusionsmodellen. Welche Rolle spielt σ_t?

---

### 3. Conditional Generation & Classifier-Free Guidance

**Frage 11:**
Was ist Classifier-Free Guidance (CFG)? Wie funktioniert es und welche zwei Vorhersagen werden kombiniert?

**Frage 12:**
Was ist der Unterschied zwischen Classifier Guidance und Classifier-Free Guidance? Warum ist CFG heute vorherrschend?

**Frage 13:**
Was ist Latent Diffusion (z.B. Stable Diffusion) und warum ist es so effizient? Welche Rolle spielt der VAE?

**Frage 14:**
Beschreiben Sie den Unterschied zwischen pixel-space und latent-space Diffusion. Was sind die Vor- und Nachteile?

**Frage 15:**
Was ist die "Guidance Scale" w und wie beeinflusst sie die generierten Bilder? Was passiert bei w = 1 vs. w > 1?

---

## Teil B: Formeln & Berechnungen (8 Fragen)

### 4. Diffusions-Formeln

**Frage 16:**
Schreiben Sie die Forward Process Formel auf, die direkt von x₀ zu x_t springt:

```
x_t = ?
```

**Frage 17:**
Schreiben Sie die DDPM Sampling-Formel auf:

```
x_{t-1} = ?
```

**Frage 18:**
Was berechnet das Modell ε_θ(x_t, t) beim Training? Schreiben Sie den MSE Loss auf.

**Frage 19:**
Schreiben Sie die Classifier-Free Guidance Formel auf:

```
ε̂_θ(x_t, t, c) = ?
```

---

### 5. Parameter und Berechnungen

**Frage 20:**
Berechnen Sie: Bei einem Forward Process mit α_t = 0.99, was ist ᾱ₁₀₀ (nach 100 Schritten)?

**Frage 21:**
Erklären Sie, wie der Noise Schedule definiert ist. Warum sollten die α_t monoton fallen?

**Frage 22:**
Was bedeutet es mathematisch, wenn ᾱ_T ≈ 0? Was ist die Konsequenz für x_T?

**Frage 23:**
Berechnen Sie den Skalierungsfaktor in der DDPM Sampling-Formel für α_t = 0.99: (1-α_t)/√(1-ᾱ_t)

---

## Teil C: Vergleiche & Analyse (10 Fragen)

### 6. Generative Modelle im Vergleich

**Frage 24:**
Vergleichen Sie GANs, VAEs und Diffusionsmodelle in einer Tabelle:

| Aspekt | GAN | VAE | Diffusion |
|--------|-----|-----|-----------|
| Bildqualität | ? | ? | ? |
| Training-Stabilität | ? | ? | ? |
| Mode Collapse | ? | ? | ? |
| Sampling-Geschwindigkeit | ? | ? | ? |
| Latent Space | ? | ? | ? |

**Frage 25:**
Warum haben Diffusionsmodelle kein Mode Collapse Problem im Gegensatz zu GANs?

**Frage 26:**
Erklären Sie, warum Diffusionsmodelle stabileres Training haben als GANs, obwohl beide generative Modelle sind.

**Frage 27:**
Was sind die Hauptnachteile von Diffusionsmodellen gegenüber GANs und VAEs? Wann würde man welches Modell bevorzugen?

---

### 7. XAI - Taxonomie und Grundlagen

**Frage 28:**
Was ist der Unterschied zwischen Interpretable ML und Explainable AI? Für wen sind diese Konzepte gedacht?

**Frage 29:**
Erklären Sie die Unterscheidung zwischen intrinsisch und post-hoc interpretierbar. Geben Sie jeweils ein Beispiel.

**Frage 30:**
Was ist der Unterschied zwischen lokalen und globalen Erklärungen? Geben Sie für beide ein Beispiel.

**Frage 31:**
Was bedeutet "modellunabhängig" (model-agnostic) im Kontext von XAI? Welche Methoden sind modellunabhängig?

---

## Teil D: Praktische Anwendungen & Edge Cases (7 Fragen)

### 8. XAI-Methoden

**Frage 32:**
Wie funktioniert Permutation Feature Importance (PFI)? Beschreiben Sie den Algorithmus Schritt für Schritt.

**Frage 33:**
Was ist LIME (Local Interpretable Model-agnostic Explanations)? Erklären Sie das Konzept des Surrogate-Modells.

**Frage 34:**
Was sind Shapley Values und woher stammen sie? Erklähen Sie die Berechnung des marginalen Beitrags.

**Frage 35:**
Was ist SHAP (SHapley Additive exPlanations)? Welche drei wichtigen Eigenschaften hat es?

**Frage 36:**
Was sind Counterfactual Explanations? Geben Sie ein konkretes Beispiel aus dem Kreditwesen.

---

### 9. Edge Cases & tiefes Verständnis

**Frage 37:**
Warum ist Permutation Feature Importance problematisch bei stark korrelierten Features? Was passiert?

**Frage 38:**
Warum ist LIME rechenintensiv? Welche Parameter müssen sorgfältig gewählt werden?

**Frage 39:**
Warum ist die Berechnung exakter Shapley Values exponentiell in der Anzahl der Features? Wie wird das in der Praxis gelöst?

**Frage 40:**
Vergleichen Sie LIME und SHAP: Welche Vor- und Nachteile hat jede Methode? Wann würden Sie welche verwenden?

---

## Antworten & Lösungen

<details>
<summary>Klicken Sie hier, um die Antworten anzuzeigen</summary>

### Teil A Antworten

**A1:** Diffusionsmodelle bestehen aus zwei Prozessen: Forward Process (Rauschen hinzufügen) und Reverse Process (Rauschen entfernen). Beim Training wird gelernt, Rauschen schrittweise zu entfernen. Bei der Generierung wird aus reinem Rauschen durch sukzessives Denoising ein Bild erzeugt.

**A2:** Beim Forward Process wird schrittweise Gauß-Rauschen zu einem Bild x₀ hinzugefügt: x_t = √(α_t)·x_{t-1} + √(1-α_t)·ε. Nach T Schritten ist das Bild fast reines Rauschen.

**A3:** Durch die geschlossene Form (closed form) des Forward Process: x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε, wobei ᾱ_t = Π_{s=1}^t α_s. Dies ermöglicht effizientes Training.

**A4:** Im Reverse Process lernt ein neuronales Netzwerk, das Rauschen schrittweise zu entfernen. Das Modell wird trainiert, das Rauschen ε vorherzusagen, das zu x_t hinzugefügt wurde.

**A5:** Direkte x₀-Vorhersage führt zu verschwommenen Bildern (Blurriness), da das Modell Mittelwerte über viele mögliche Bilder bildet. Die Vorhersage von ε ist stabiler und führt zu schärferen Ergebnissen.

**A6:** DDPM (Denoising Diffusion Probabilistic Models) Sampling:
1. Starte mit x_T ~ N(0, I)
2. Für t = T, T-1, ..., 1:
   - Berechne ε_θ(x_t, t)
   - Berechne x_{t-1} mit DDPM-Formel
3. Gebe x₀ aus

**A7:** DDPM benötigt typischerweise 1000 Schritte. Weniger Schritte führen zu schlechterer Qualität, weil die Annahme kleiner Schritte in der Herleitung verletzt wird.

**A8:** DDIM (Denoising Diffusion Implicit Models) ist eine deterministische Variante. Der Hauptvorteil: Erlaubt das Überspringen von Schritten (z.B. nur 50 Schritte) mit fast gleicher Qualität.

**A9:** DDIM ist deterministisch (kein zufälliges Rauschen bei σ_t = 0), daher können Schritte nicht-stochastisch übersprungen werden. DDPM ist stochastisch und jeder Schritt hängt vom vorherigen ab.

**A10:** Stochastisches Sampling hat Zufallsrauschen (σ_t > 0), erzeugt verschiedene Bilder bei gleichem Start. Deterministisches Sampling (σ_t = 0) erzeugt reproduzierbare Ergebnisse.

**A11:** CFG kombiniert bedingte und unbedingte Vorhersagen: ε̂ = ε(x,t,∅) + w·(ε(x,t,c) - ε(x,t,∅)). Dabei ist ∅ unbedingt (keine Klasse).

**A12:** Classifier Guidance benötigt einen separaten Classifier auf verrauschten Bildern. CFG trainiert das Modell abwechselnd mit und ohne Konditionierung, kein separater Classifier nötig.

**A13:** Latent Diffusion führt Diffusion im komprimierten VAE-Latent-Space durch. Der VAE kodiert Bilder in einen kleineren Raum, was effizienter ist für hohe Auflösungen.

**A14:** Pixel-space: Diffusion direkt auf Pixeln (langsam, speicherintensiv). Latent-space: Diffusion im komprimierten Raum (schneller, effizienter, kann Artefakte haben).

**A15:** w (Guidance Scale) steuert die Stärke der Konditionierung. Bei w=1: Standard-Diffusion. Bei w>1: Stärkere Einhaltung der Konditionierung, oft bessere Qualität aber weniger Diversität.

### Teil B Antworten

**A16:**
```
x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε
```

**A17:**
```
x_{t-1} = (1/√α_t)·(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ(x_t, t)) + σ_t·z
```

**A18:** Das Modell lernt das Rauschen ε vorherzusagen:
```
L = ||ε - ε_θ(x_t, t)||²
```

**A19:**
```
ε̂_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + w·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```

**A20:** ᾱ₁₀₀ = 0.99¹⁰⁰ ≈ 0.366. Der Signalanteil beträgt √0.366 ≈ 60.5%, der Rauschanteil √0.634 ≈ 79.6%. (Hinweis: Das Bild ist eine Mischung aus Signal und Rauschen, nicht nur "37% Originalbild".)

**A21:** Der Noise Schedule definiert α_t für jeden Zeitschritt. α_t sollten monoton fallen, um zunehmend mehr Rauschen hinzuzufügen. Typisch linear oder cosinus-förmig.

**A22:** ᾱ_T ≈ 0 bedeutet x_T ≈ √(1-ᾱ_T)·ε ≈ ε (reines Rauschen). Das Bild ist vollständig zerstört.

**A23:** Bei α_t = 0.99: ᾱ_t ≈ 0.366 (für t=100). Faktor = (1-0.99)/√(1-0.366) = 0.01/0.796 ≈ 0.0126

### Teil C Antworten

**A24:**
| Aspekt | GAN | VAE | Diffusion |
|--------|-----|-----|-----------|
| Bildqualität | Sehr hoch | Verwaschen | Sehr hoch |
| Training-Stabilität | Instabil | Stabil | Sehr stabil |
| Mode Collapse | Ja | Nein | Nein |
| Sampling-Geschwindigkeit | Schnell (1 Schritt) | Schnell (1 Schritt) | Langsam (100-1000 Schritte) |
| Latent Space | Unklar | Strukturiert | Gut strukturiert |

**A25:** Diffusionsmodelle haben keinen adversarialen Trainingsprozess. Das Lernen der Reverse-Distribution deckt alle Datenmoden gleichmäßig ab, ohne dass ein Discriminator bestimmte Modi unterdrückt.

**A26:** GANs haben ein Minimax-Spiel mit instabilem Gleichgewicht. Diffusionsmodelle haben einen einfachen MSE-Loss auf Rauschen-Vorhersage, keine adversarialen Komponenten.

**A27:** Hauptnachteile: Langsames Sampling (viele Schritte), hoher Rechenaufwand. GANs bevorzugt wenn Geschwindigkeit wichtig ist, Diffusion wenn Qualität/Stabilität wichtig ist.

**A28:** Interpretable ML: Verständlichkeit für Entwickler/ML-Experten. Explainable AI: Allgemeinverständliche Erklärungen auch für Nicht-Experten/Laien.

**A29:** Intrinsisch: Modell ist von Natur aus interpretierbar (z.B. lineare Regression, Entscheidungsbäume). Post-hoc: Erklärung wird nachträglich erstellt (z.B. LIME, SHAP).

**A30:** Lokal: Erklärung einzelner Vorhersagen (z.B. warum wurde dieser Kredit abgelehnt?). Global: Verständnis des Gesamtverhaltens (z.B. welche Features sind generell wichtig?).

**A31:** Modellunabhängig: Funktioniert mit beliebigen Modellen, nutzt keine Interna. Beispiele: LIME, SHAP, PFI. Gegenstück: Modellspezifisch (z.B. Attention-Visualisierung bei Transformern).

### Teil D Antworten

**A32:** PFI Algorithmus:
1. Miss Baseline-Performance auf Testdaten
2. Permutiere (mische) Werte eines Features zufällig
3. Miss Performance erneut
4. Wichtigkeit = Performance-Abfall
5. Wiederhole für alle Features

**A33:** LIME stört den zu erklärenden Punkt, erzeugt synthetische Nachbarschaftsdaten, trainiert ein einfaches Surrogate-Modell (z.B. lineare Regression) auf diesen Daten.

**A34:** Shapley Values stammen aus kooperativer Spieltheorie. Marginaler Beitrag: v(S∪{i}) - v(S) = Vorhersage mit Feature i minus Vorhersage ohne Feature i.

**A35:** SHAP basiert auf Shapley Values und hat drei Eigenschaften:
- **Additiv:** Summe der SHAP-Werte + Baseline = Vorhersage
- **Konsistent:** Stabile Erklärungen
- **Fair:** Berücksichtigt alle Feature-Kombinationen

**A36:** Counterfactuals zeigen minimale Änderungen, die zur gewünschten Vorhersage führen. Beispiel: "Wenn das Einkommen 5% höher wäre, wäre der Kredit genehmigt worden."

**A37:** Bei korrelierten Features wird die Wichtigkeit zwischen den korrelierten Features aufgeteilt. Einzeln sind sie weniger wichtig, zusammen sehr wichtig.

**A38:** LIME ist rechenintensiv weil für jede Erklärung ein neues Surrogate-Modell trainiert werden muss. Parameter: Größe der Nachbarschaft (π_x), Komplexität des Surrogats (Ω).

**A39:** Exakte Shapley-Werte benötigen 2^n Berechnungen für n Features. In der Praxis: Approximationen (Sampling), TreeSHAP (exakt für Bäume), KernelSHAP (modellunabhängig).

**A40:** LIME: Schneller, approximativ, flexibel. SHAP: Theoretisch fundierter, konsistent, aber rechenaufwändiger. LIME für schnelle Erklärungen, SHAP für präzise, faire Erklärungen.

</details>

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 36-40 | 🟢 Exzellent - Bereit für die Prüfung |
| 30-35 | 🟢 Gut - Kleine Wiederholung empfohlen |
| 24-29 | 🟡 Befriedigend - Themen wiederholen |
| 18-23 | 🟡 Ausreichend - Tag 5 wiederholen |
| <18 | 🔴 Nachholbedarf - Zusammenfassungen nochmal lesen |

---

## Wichtige Formeln (auswendig lernen!)

**Forward Process (Closed Form):**
```
x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε
```

**Diffusion Training Loss:**
```
L = ||ε - ε_θ(x_t, t)||²
```

**DDPM Sampling:**
```
x_{t-1} = (1/√α_t)·(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ(x_t, t)) + σ_t·z
```

**Classifier-Free Guidance:**
```
ε̂_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + w·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```

**Shapley Value:**
```
φ_i = Σ_{S⊆N\{i}} (|S|!(|N|-|S|-1)! / |N|!) × [v(S∪{i}) - v(S)]
```

**LIME Optimierung:**
```
g = argmin_{g∈G} L(f, g, π_x) + Ω(g)
```

---

**Viel Erfolg!** 🎯
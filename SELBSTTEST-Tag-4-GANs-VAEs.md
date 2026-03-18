# Selbsttest Tag 4: GANs, VAEs & Generative Modelle

**Umfang:** Ausführlicher Test zu allen Themen des Tages  
**Zeitansatz:** 50-70 Minuten  
**Hinweis:** Antworten nicht einfach ablesen - erst selbst überlegen, dann nachschlagen!

---

## Teil A: Grundkonzepte & Verständnis (12 Fragen)

### 1. GAN-Grundkonzept

**Frage 1:**  
Erklären Sie das adversarielle Training bei GANs. Welche zwei Netzwerke gibt es und wie spielen sie gegeneinander?

**Frage 2:**  
Was ist der Input und Output des Generators? Was ist sein Ziel?

**Frage 3:**  
Was ist der Input und Output des Discriminators? Was ist sein Ziel?

**Frage 4:**  
Warum wird der Generator nicht direkt auf echten Daten trainiert? Warum braucht er den Discriminator?

---

### 2. GAN Training & Loss

**Frage 5:**  
Schreiben Sie die Binary Cross Entropy (BCE) Loss-Formel für GANs auf. Was versucht der Discriminator zu maximieren/minimieren? Was versucht der Generator?

**Frage 6:**  
Beschreiben Sie die Trainings-Schleife eines GANs Schritt für Schritt. In welcher Reihenfolge werden Generator und Discriminator trainiert?

**Frage 7:**  
Was ist Mode Collapse? Beschreiben Sie das Problem und warum es auftritt.

---

### 3. Wasserstein GAN (WGAN)

**Frage 8:**  
Was sind die zwei Hauptprobleme bei GANs mit BCE Loss, die WGAN löst?

**Frage 9:**  
Was ist der Unterschied zwischen einem Discriminator und einem Critic? Was gibt jeder aus?

**Frage 10:**  
Was bedeutet 1-Lipschitz-Stetigkeit und warum wird sie bei WGAN benötigt? Nennen Sie zwei Methoden, um sie zu erzwingen.

---

### 4. Conditional GAN & Controllable Generation

**Frage 11:**  
Was ist der Unterschied zwischen einem normalen GAN und einem Conditional GAN (cGAN)? Wie wird die Klasseninformation übergeben?

**Frage 12:**  
Was ist das Entanglement-Problem bei Controllable Generation? Geben Sie ein konkretes Beispiel.

---

## Teil B: Formeln & Berechnungen (10 Fragen)

### 5. GAN Evaluation

**Frage 13:**  
Schreiben Sie die Formel für den Inception Score (IS) auf. Was bedeuten p(y|x) und p(y)?

**Frage 14:**  
Schreiben Sie die Formel für die Fréchet Inception Distance (FID) auf. Was bedeuten μ und Σ?

**Frage 15:**  
Interpretieren Sie: Ein GAN erreicht IS = 1.2 und FID = 250. Ein anderes erreicht IS = 8.5 und FID = 15. Welches ist besser und warum?

---

### 6. VAE Grundlagen

**Frage 16:**  
Warum kann ein normaler Autoencoder nicht sinnvoll zur Generierung verwendet werden? Was ist das Problem mit dem Latent Space?

**Frage 17:**  
Erklären Sie den Reparametrisierungs-Trick. Warum ist er notwendig? Schreiben Sie die Formel auf.

**Frage 18:**  
Was gibt der VAE-Encoder aus? Wie unterscheidet sich das vom normalen Autoencoder?

---

### 7. ELBO & KL-Divergenz

**Frage 19:**  
Schreiben Sie die ELBO (Evidence Lower Bound) Loss-Formel für VAEs auf. Was sind die zwei Komponenten?

**Frage 20:**  
Schreiben Sie die KL-Divergenz für zwei Gauß-Verteilungen auf:
D_KL(N(μ, σ²) || N(0, I)) = ?

**Frage 21:**  
Berechnen Sie die KL-Divergenz für μ = [0.5, -0.3] und σ = [1.2, 0.8].

**Frage 22:**  
Ein VAE hat Reconstruction Loss = 45.2 und KL-Divergenz = 12.8. Was ist der Gesamt-ELBO Loss?

---

## Teil C: Vergleiche & Analyse (10 Fragen)

### 8. GAN vs VAE

**Frage 23:**  
Vergleichen Sie GAN und VAE in einer Tabelle:
- Training-Stabilität
- Bildqualität
- Latent Space Struktur
- Mode Collapse

**Frage 24:**  
Warum produzieren VAEs typischerweise "verwaschene" Bilder, während GANs schärfere Bilder erzeugen?

**Frage 25:**  
Warum ist der Latent Space bei VAEs strukturierter als bei GANs? Welche Eigenschaft des VAE-Loss sorgt dafür?

---

### 9. WGAN Varianten

**Frage 26:**  
Vergleichen Sie Weight Clipping vs Gradient Penalty bei WGAN:
- Wie funktioniert jede Methode?
- Welche ist bevorzugt und warum?

**Frage 27:**  
Was ist die Earth Mover's Distance (EMD)? Warum ist sie besser geeignet als BCE für GANs?

---

### 10. Evaluation Metriken

**Frage 28:**  
Was sind die zwei Hauptkriterien für die Evaluation von generativen Modellen? Erklären Sie jedes kurz.

**Frage 29:**  
Warum ist FID typischerweise besser als Inception Score? Was ist der Hauptnachteil von IS?

**Frage 30:**  
Was bedeutet "niedriger FID = besser"? Was misst FID konkret?

---

### 11. Autoencoder vs VAE

**Frage 31:**  
Vergleichen Sie Autoencoder und VAE:
- Encoder Output
- Latent Space Eigenschaften
- Eignung für Generierung

**Frage 32:**  
Warum wird beim VAE eine Standardnormalverteilung N(0, I) als Prior p(z) verwendet?

---

## Teil D: Praktische Anwendungen & Edge Cases (8 Fragen)

### 12. Training & Praxis

**Frage 33:**  
Warum sollte der Discriminator typischerweise öfter trainiert werden als der Generator (z.B. 5:1 Verhältnis)?

**Frage 34:**  
Was passiert, wenn der Discriminator "zu gut" wird? Wie äußert sich das im Training?

**Frage 35:**  
Wie kann man Mode Collapse in der Praxis erkennen? Nennen Sie zwei Anzeichen.

**Frage 36:**  
Was ist der Unterschied zwischen Conditional GAN und Controllable Generation? Geben Sie jeweils ein Anwendungsbeispiel.

---

### 13. Grenzfälle & tiefes Verständnis

**Frage 37:**  
Warum funktioniert der Reparametrisierungs-Trick? Welches Problem würde ohne ihn beim Backpropagation entstehen?

**Frage 38:**  
Was passiert, wenn die KL-Divergenz im VAE zu stark gewichtet wird? Was passiert, wenn sie zu schwach gewichtet wird?

**Frage 39:**  
Warum kann WGAN auch dann noch trainieren, wenn die Verteilungen von echten und generierten Daten nicht überlappen?

**Frage 40:**  
Ein VAE generiert unscharfe Bilder, obwohl der Reconstruction Loss niedrig ist. Was könnte die Ursache sein und wie könnte man das beheben?

---

## Antworten & Lösungen

<details>
<summary>Klicken Sie hier, um die Antworten anzuzeigen</summary>

### Teil A Antworten

**A1:** Zwei Netzwerke: Generator (erzeugt Daten) und Discriminator (unterscheidet echt/fake). Adversarial: Generator versucht Discriminator zu täuschen, Discriminator versucht Fälschungen zu erkennen. Sie spielen ein Minimax-Spiel.

**A2:** Generator Input: Random Noise z (z.B. aus N(0,I)). Output: Generierte Daten (z.B. Bild). Ziel: Verteilung P(X) der echten Daten lernen.

**A3:** Discriminator Input: Echte oder generierte Daten. Output: Wahrscheinlichkeit "echt" (0=fake, 1=real). Ziel: P(Y|X) - Wahrscheinlichkeit für Klasse Y gegeben X.

**A4:** Generator hat keinen Zugriff auf echte Daten. Er muss lernen, welche Daten "realistisch" sind - das Feedback kommt vom Discriminator. Ohne Discriminator hätte er keine Loss-Signal für Qualität.

**A5:** L = -[y·log(D(x)) + (1-y)·log(1-D(G(z)))]. Discriminator: maximiert Loss (erkennt echt/fake korrekt). Generator: minimiert log(1-D(G(z))) oder maximiert log(D(G(z))) (täuscht Discriminator).

**A6:** 1) Discriminator trainieren: Generiere Fake, klassifiziere Real+Fake, update D. 2) Generator trainieren: Generiere Fake, D klassifiziert, update G (maximiert D-Error). Abwechselnd wiederholen.

**A7:** Mode Collapse: Generator produziert nur eine Mode (z.B. nur eine Ziffer bei MNIST). Ursache: Generator findet eine Schwäche im Discriminator und nutzt sie aus, statt die ganze Verteilung zu lernen.

**A8:** 1) Wenn Discriminator zu gut: Gradienten verschwinden. 2) BCE funktioniert nicht gut ohne Überlapp der Verteilungen.

**A9:** Discriminator gibt Wahrscheinlichkeit aus (0-1, Sigmoid). Critic gibt Score aus (unbeschränkt, linear). Critic misst Earth Mover's Distance.

**A10:** 1-Lipschitz: |∇C(x)| ≤ 1 überall (Steigung beschränkt). Methoden: Weight Clipping (Gewichte auf [-c,c] beschränken) oder Gradient Penalty (λ·E[(||∇C(x̂)|| - 1)²] zum Loss hinzufügen).

**A11:** cGAN erhält zusätzliche Klasseninformation c als Input. Generator: [z | c] → Bild. Discriminator: [x | c] → Real/Fake. Ermöglicht gezielte Generierung bestimmter Klassen.

**A12:** Entanglement: Eine Dimension im Latent Space beeinflusst mehrere Eigenschaften. Beispiel: "Brille"-Dimension ändert auch "Bart" - die Attribute sind vermischt/verschränkt.

### Teil B Antworten

**A13:** IS = exp(E_x[KL(p(y|x) || p(y))]). p(y|x): Klassenwahrscheinlichkeit für Bild x (hohe Entropie = schlecht). p(y): Marginalverteilung über alle Bilder (gleichverteilt = gut).

**A14:** FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2·√(Σ_real·Σ_fake)). μ: Mittelwert der Features. Σ: Kovarianzmatrix der Features.

**A15:** Zweites GAN ist besser (IS=8.5, FID=15). IS soll hoch sein (klare Klassifikation, diverse Klassen). FID soll niedrig sein (geringe Distanz zwischen echten und generierten Verteilungen).

**A16:** Normaler Autoencoder: Latent Space hat viele "leere" Bereiche ohne garantierte Verteilung. Keine kontinuierliche Struktur → schwer zu sampeln. VAE erzwingt strukturierten Latent Space durch KL-Divergenz.

**A17:** Reparametrisierungs-Trick: z = μ + σ ⊙ ε, wobei ε ~ N(0,I). Notwendig, um Backpropagation durch zufällige Sampling-Operation zu ermöglichen (ε ist zufällig, aber z bleibt differenzierbar bezüglich μ und σ).

**A18:** VAE-Encoder gibt μ(x) und σ(x) aus (Parameter einer Normalverteilung). Normaler Autoencoder gibt direkt z aus. VAE: z ~ N(μ(x), diag(σ²(x))).

**A19:** ELBO = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z)). Komponenten: 1) Reconstruction Loss (E_q[log p_θ(x|z)]), 2) KL-Divergenz (Regularisierung).

**A20:** D_KL(N(μ, σ²) || N(0, I)) = 0.5 · Σ(μ² + σ² - log(σ²) - 1)

**A21:** D_KL = 0.5 · [(0.25 + 1.44 - log(1.44) - 1) + (0.09 + 0.64 - log(0.64) - 1)] = 0.5 · [(0.69 - 0.365) + (-0.27 + 0.446)] = 0.5 · [0.325 + 0.176] = 0.5 · 0.501 = 0.2505

**A22:** ELBO = Reconstruction Loss + KL-Divergenz = 45.2 + 12.8 = 58.0 (oder -ELBO = -45.2 - 12.8 = -58.0, je nach Vorzeichenkonvention)

### Teil C Antworten

**A23:**
| Aspekt | GAN | VAE |
|--------|-----|-----|
| Training-Stabilität | Instabil (Mode Collapse) | Stabil |
| Bildqualität | Scharf | Verwaschen |
| Latent Space | Weniger strukturiert | Strukturiert (N(0,I)) |
| Mode Collapse | Ja (Problem) | Nein |

**A24:** VAE optimiert Likelihood direkt, was zu Mittelwert-Bildern führt (Blur). GAN: Discriminator erzwingt Realismus, führt zu schärferen Details.

**A25:** VAE erzwingt durch KL-Divergenz-Term, dass q(z|x) nah an Prior p(z)=N(0,I) bleibt. Das strukturiert den Latent Space und ermöglicht sinnvolles Sampling.

**A26:** Weight Clipping: Gewichte auf [-c,c] beschränken. Einfach, aber kann Kapazität einschränken. Gradient Penalty: Loss-Term mit λ·E[(||∇C(x̂)|| - 1)²]. Bevorzugt: GP, weil weicher und effektiver.

**A27:** EMD = minimale "Arbeit" um eine Verteilung in die andere zu transformieren. Besser als BCE, weil sie auch ohne Überlapp definiert ist und glattere Gradienten liefert.

**A28:** Fidelity (Qualität einzelner Bilder - Realitätsnähe) und Diversity (Vielfalt der generierten Bilder - keine Wiederholungen, alle Modi abgedeckt).

**A29:** FID betrachtet echte UND generierte Bilder und misst Verteilungsähnlichkeit. IS betrachtet nur generierte Bilder und verlässt sich auf Classifier-Qualität. FID ist robuster.

**A30:** FID misst Distanz zwischen Verteilungen von echten und generierten Bildern im Feature-Space (Inception-Netz). Niedriger = Verteilungen sind ähnlicher.

**A31:** Autoencoder: Encoder gibt direkt z aus, Latent Space unstrukturiert, nicht für Generierung geeignet. VAE: Encoder gibt μ, σ aus, Latent Space strukturiert (N(0,I)), geeignet für Generierung.

**A32:** N(0,I) ist einfach zu sampeln und hat gute Eigenschaften (kontinuierlich, keine Lücken). KL-Divergenz zieht q(z|x) zu diesem Prior, ermöglicht sinnvolle Interpolation.

### Teil D Antworten

**A33:** Discriminator muss stark genug sein, um nützliche Gradienten zu liefern. Wenn G zu schnell lernt, überholt er D und Gradienten verschwinden. D muss "voraus" bleiben.

**A34:** Discriminator erkennt alle Fakes perfekt (D(G(z)) ≈ 0). Generator bekommt keine Gradienten mehr (log(1-D(G(z))) saturates). Training bricht zusammen.

**A35:** Anzeichen: 1) Generator produziert identische oder sehr ähnliche Bilder. 2) Inception Score wird niedrig (niedrige Diversity). 3) FID wird hoch trotz guter Einzelbildqualität.

**A36:** Conditional GAN: Klasse wird als Input gegeben (z.B. "generiere Ziffer 7"). Controllable Generation: Steuerung durch Manipulation im Latent Space (z.B. "mache das Bild älter" durch Verschiebung in z).

**A37:** Trick ermöglicht Gradientenfluss durch zufällige Variable. Ohne ihn: Sampling-Operation nicht differenzierbar → Backpropagation würde bei z stoppen, μ und σ könnten nicht gelernt werden.

**A38:** Zu stark: Latent Space wird zu sehr auf N(0,I) gezwungen, Reconstruction wird schlecht (unscharfe Bilder). Zu schwach: Latent Space verliert Struktur, ähnlich wie normaler Autoencoder.

**A39:** EMD (Wasserstein Loss) ist auch für nicht-überlappende Verteilungen definiert und liefert sinnvolle Gradienten. BCE würde versagen (keine Überlapp = keine Gradienten).

**A40:** Ursache: KL-Divergenz zu stark gewichtet oder Prior zu stark erzwungen. Lösung: β-VAE (Gewichtung der KL-Divergenz anpassen) oder KL-Annealing (Gewicht langsam erhöhen).

</details>

---

## Bewertungsschlüssel

| Richtige Antworten | Bewertung |
|-------------------|-----------|
| 36-40 | 🟢 Exzellent - Bereit für Tag 5 |
| 30-35 | 🟢 Gut - Kleine Wiederholung empfohlen |
| 24-29 | 🟡 Befriedigend - Themen wiederholen |
| 18-23 | 🟡 Ausreichend - Tag 4 wiederholen |
| <18 | 🔴 Nachholbedarf - Zusammenfassung nochmal lesen |

---

## Wichtige Formeln (auswendig lernen!)

**GAN Binary Cross Entropy:**
```
L = -[y·log(D(x)) + (1-y)·log(1-D(G(z)))]
```

**Wasserstein Loss:**
```
min_G max_C E[C(x)] - E[C(G(z))]
```

**Gradient Penalty:**
```
GP = λ · E[(||∇C(x̂)|| - 1)²]
```

**Inception Score:**
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```

**Fréchet Inception Distance:**
```
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2·√(Σ_real·Σ_fake))
```

**VAE Reparametrisierungs-Trick:**
```
z = μ + σ ⊙ ε,  wobei ε ~ N(0, I)
```

**VAE ELBO Loss:**
```
L = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

**KL-Divergenz (Gauss):**
```
D_KL(N(μ, σ²) || N(0, I)) = 0.5 · Σ(μ² + σ² - log(σ²) - 1)
```

---

**Viel Erfolg!** 🎯

# ZUSAMMENFASSUNG 05: GANs, VAEs und Diffusionsmodelle

## Übersicht
- Seitenzahl: ~90 Seiten
- Hauptthemen: GANs (Training, Conditional/Controllable, Evaluation), VAEs, Diffusionsmodelle

## Detaillierte Inhalte

### TEIL 1: GENERATIVE ADVERSARIAL NETWORKS (GANs)

#### 1.1 GAN-Grundkonzept

**Zwei Netzwerke:**
- **Generator:** Erzeugt realistische Daten (z.B. Bilder)
- **Discriminator:** Unterscheidet echt von gefälscht

**Adversarial Training:**
- Zwei Netzwerke spielen gegeneinander
- Generator versucht Discriminator auszutricksen
- Discriminator versucht Fälschungen zu erkennen
- Abwechselnde Optimierung

**Generator:**
- Input: Random Noise (z-Vektor)
- Output: Generierte Daten (z.B. Bild)
- Ziel: Verteilung P(X) der echten Daten lernen

**Discriminator:**
- Input: Echte oder generierte Daten
- Output: Wahrscheinlichkeit "echt" (0=fake, 1=real)
- Ziel: P(Y|X) - Wahrscheinlichkeit für Klasse Y gegeben X

#### 1.2 GAN Training

**Binary Cross Entropy (BCE) Loss:**
```
L = -[y·log(D(x)) + (1-y)·log(1-D(G(z)))]
```
- **Discriminator:** maximiert Loss (erkennt echt/fake)
- **Generator:** minimiert Loss (täuscht Discriminator)

**Trainings-Schleife:**
1. Discriminator-Training:
   - Generiere Fake-Bild mit Generator
   - Discriminator klassifiziert Real und Fake
   - Update Discriminator-Parameter
2. Generator-Training:
   - Generiere Fake-Bild
   - Discriminator klassifiziert
   - Update Generator-Parameter (maximiert Discriminator-Loss)

**Problem: Mode Collapse**
- Generator produziert nur eine Mode (z.B. nur eine Ziffer)
- Lösung: Wasserstein GAN

#### 1.3 Wasserstein GAN (WGAN)

**Probleme mit BCE:**
- Wenn Discriminator zu gut: Gradienten verschwinden
- Kein gutes Feedback für Generator

**Wasserstein Loss:**
```
min_G max_C E[C(x)] - E[C(G(z))]
```
- **C:** Critic (anstatt Discriminator) - gibt Score aus (keine Wahrscheinlichkeit)
- Misst Earth Mover's Distance (EMD)
- Funktioniert auch ohne Überlapp der Verteilungen

**1-Lipschitz-Stetigkeit:**
- |∇C(x)| ≤ 1 überall
- **Weight Clipping:** Gewichte auf [-c, c] beschränken
- **Gradient Penalty (bevorzugt):** Loss-Term mit λ·E[(||∇C(x̂)|| - 1)²]
  - x̂ = Interpolation zwischen real und fake

#### 1.4 Conditional GAN

**Unterschied zum normalen GAN:**
- Zusätzliche Klassen-Information als Input
- Generator erzeugt Bilder spezifischer Klasse
- Discriminator prüft Echtheit UND Klasse

**Input:**
```
Generator: [Noise z | Class c] → Generated Image
Discriminator: [Image x | Class c] → Real/Fake Score
```

#### 1.5 Controllable Generation

**Idee:**
- Änderung des z-Vektors steuert Eigenschaften
- Interpolation im z-Raum zwischen zwei Bildern
- Richtungen im z-Raum entsprechen Attributen

**Entanglement-Problem:**
- Eine Dimension beeinflusst mehrere Eigenschaften
- Z.B. Brille ↔ Bart korreliert

**Lösungen:**
1. **Disentanglement forcieren:** Spezieller Regularisierungsterm
2. **Classifier Gradients:** Vortrainierter Classifier findet Richtungen

#### 1.6 GAN Evaluation

**Zwei Hauptkriterien:**
- **Fidelity:** Qualität einzelner Bilder (Realitätsnähe)
- **Diversity:** Vielfalt der generierten Bilder (kein Mode Collapse)

**Metriken:**

**Inception Score (IS):**
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```
- p(y|x): Klassenwahrscheinlichkeit für Bild x (hohe Entropie = schlecht)
- p(y): Marginalverteilung über alle Bilder (gleichverteilt = gut)
- Nachteile: Betrachtet nur generierte Bilder, verlässt sich auf Classifier

**Fréchet Inception Distance (FID):**
```
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2·√(Σ_real·Σ_fake))
```
- Embeddings von echten und generierten Bildern
- Annahme: multivariate Normalverteilung
- **Niedriger FID = besser**

---

### TEIL 2: VARIATIONAL AUTOENCODER (VAE)

#### 2.1 Autoencoder-Grundlagen

**Struktur:**
- **Encoder:** Komprimiert Input in Latent-Space
- **Decoder:** Rekonstruiert Input aus Latent-Representation

**Loss:**
```
L = ||x - x̂||²  (Reconstruction Loss)
```

**Problem für Generierung:**
- Latent Space hat viele "leere" Bereiche
- Keine garantierte Verteilung → schwer zu sampeln

#### 2.2 VAE-Konzept

**Encoder:**
- Gibt nicht direkt z aus, sondern μ(x) und σ(x)
- Latent Variable z ~ N(μ(x), diag(σ²(x)))

**Reparametrisierungs-Trick:**
```
z = μ + σ ⊙ ε,  wobei ε ~ N(0, I)
```
- ε ist zufällig, aber z bleibt differenzierbar

**Decoder:**
- Rekonstruiert x aus z
- p_θ(x|z) ~ exp(-||x - d_θ(z)||²/τ)

#### 2.3 VAE Loss (ELBO)

**Evidence Lower Bound:**
```
L = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

**Zwei Komponenten:**
1. **Reconstruction Loss:** ||x - x̂||² (Wie gut wird rekonstruiert?)
2. **KL-Divergence:** D_KL(q(z|x) || p(z)) (Wie nah ist q an Prior p(z)?)

**KL-Divergence für Gauss:**
```
D_KL(N(μ, σ²) || N(0, I)) = 0.5 · Σ(μ² + σ² - log(σ²) - 1)
```

**Prior p(z):**
- Typischerweise Standardnormalverteilung N(0, I)
- Erzwingt strukturierten Latent Space

#### 2.4 VAE vs GAN

| Aspekt | VAE | GAN |
|--------|-----|-----|
| Training | Stabil | Instabil (Mode Collapse) |
| Bildqualität | Verwaschen | Scharf |
| Latent Space | Strukturiert | Weniger klar |
| Training | Einfacher | Komplexer |

---

### TEIL 3: DIFFUSIONSMODELLE

#### 3.1 Grundkonzept

**Zwei Prozesse:**
1. **Forward Process:** Rauschen wird schrittweise hinzugefügt
2. **Reverse Process:** Rauschen wird schrittweise entfernt (generiert Daten)

**Vorteile:**
- Sehr hohe Bildqualität
- Stabiles Training
- Kein Mode Collapse

#### 3.2 Forward Process (Diffusion)

**Zerstört Bild x₀ durch sukzessive Addition von Gauß-Rauschen:**
```
x_t = √(α_t)·x_{t-1} + √(1-α_t)·ε,  ε ~ N(0, I)
```

**Closed Form (direkt von x₀ zu x_t):**
```
x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε
```
- ᾱ_t = Π_{s=1}^t α_s
- Bei T groß: x_T ≈ reines Rauschen ~ N(0, I)

**Noise Schedule:**
- α_t monoton fallend
- Typisch: T = 1000 Schritte

#### 3.3 Reverse Process (Denoising)

**Lerne neuronales Netzwerk ε_θ(x_t, t):**
- Vorhersage des Rauschens in x_t

**Training Loss (MSE):**
```
L = E[||ε - ε_θ(x_t, t)||²]
```
- ε: tatsächliches Rauschen
- ε_θ: Modellvorhersage

**Sampling-Algorithmus (DDPM):**
```
x_{t-1} = (1/√α_t)·(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ(x_t, t)) + σ_t·z
```
- σ_t: Std-Abweichung des Rauschens in Schritt t
- z ~ N(0, I): Zufälliges Rauschen

#### 3.4 DDIM (Schnelleres Sampling)

**Deterministische Variante:**
```
x_{t-1} = √(ᾱ_{t-1})·(x_t - √(1-ᾱ_t)·ε_θ(x_t, t))/√ᾱ_t + √(1-ᾱ_{t-1} - σ_t²)·ε_θ(x_t, t) + σ_t·z
```

**Mit σ_t = 0:** Deterministisch, erlaubt Überspringen von Schritten
- Nur 10-50 Schritte statt 1000
- Fast gleiche Qualität

#### 3.5 Conditional Generation

**Ziel:** Kontrolle über generierte Inhalte

**Möglichkeiten:**

1. **Condition als Input:** ε_θ(x_t, t, c)
   - c: Klasse, Textembedding, etc.

2. **Classifier Guidance:**
   ```
   ∇_x log p(x|c) = ∇_x log p(c|x) + ∇_x log p(x)
   ```
   - p(c|x): Vortrainierter Classifier auf verrauschten Bildern
   - s: Guidance Scale (Stärke des Einflusses)

3. **Classifier-Free Guidance (CFG):**
   ```
   ε̂_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + w·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
   ```
   - ∅: Unconditioned (zufälliges Weglassen von c während Training)
   - w: Guidance Scale
   - Kein separater Classifier nötig

#### 3.6 Netzwerk-Architektur

**Typisch: U-Net**
- Encoder-Decoder Struktur
- Skip-Connections
- Zeit-Embedding (analog zu Positional Encoding)
- Self-Attention in tieferen Layern
- Cross-Attention für Text-Konditionierung

**Latent Diffusion:**
- Diffusion im komprimierten Latent Space (VAE)
- Viel effizienter für hohe Auflösungen
- Beispiele: Stable Diffusion, Kandinsky

#### 3.7 Vor- und Nachteile

**Vorteile:**
- Hohe Qualität
- Stabiles Training
- Kein Mode Collapse
- Flexible Konditionierung

**Nachteile:**
- Langsames Sampling (viele Schritte nötig)
- Hoher Rechenaufwand

## Besonders ausführlich behandelt (wahrscheinlich prüfungsrelevant)

### ✅ GAN Training und Loss
- Warum: Grundverständnis generativer Modelle
- Was: Generator/Discriminator, BCE Loss, Alternating Training

### ✅ Wasserstein GAN
- Warum: Löst Mode Collapse und verschwindende Gradienten
- Was: Critic, EMD, Gradient Penalty, 1-Lipschitz

### ✅ VAE (Reparametrisierungs-Trick, ELBO)
- Warum: Wichtige Alternative zu GANs
- Was: Encoder/Decoder, KL-Divergenz, Reconstruction Loss

### ✅ Diffusionsmodelle (Forward/Reverse)
- Warum: State-of-the-Art in Bildgenerierung
- Was: Noise Schedule, Training Loss, DDPM Sampling

### ✅ GAN Evaluation (FID)
- Warum: Qualitätsbewertung wichtig
- Was: Inception Score, Fréchet Inception Distance

## Formeln/Algorithmen (wichtig)

### GAN BCE Loss
```
L_D = -[log(D(x)) + log(1-D(G(z)))]
L_G = -log(D(G(z)))
```

### Wasserstein Loss
```
min_G max_C E[C(x)] - E[C(G(z))]
```

### VAE ELBO
```
L = E[||x - d(z)||²] + D_KL(q(z|x) || p(z))
```

### Diffusion Forward
```
x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε
```

### Diffusion Training Loss
```
L = ||ε - ε_θ(x_t, t)||²
```

### Diffusion Sampling (DDPM)
```
x_{t-1} = (x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ)/√α_t + σ_t·z
```

### Classifier-Free Guidance
```
ε̂ = ε_θ(x_t, t, ∅) + w·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```

### FID
```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·√(Σ₁·Σ₂))
```

## Eigene Notizen/Verständnis

### 🔑 Kernpunkte
- **GANs:** Zwei Netzwerke im Wettbewerb, instabiles Training
- **WGAN:** Critic statt Discriminator, Gradient Penalty für Stabilität
- **VAE:** Probabilistischer Ansatz, KL-Regularisierung, stabileres Training
- **Diffusion:** Schrittweise Denoising, sehr hohe Qualität, aber langsam

### ⚠️ Häufige Fehler
- GAN: Mode Collapse nicht beachten
- VAE: Reparametrisierungs-Trick vergessen
- Diffusion: Warum nicht direkt x₀ vorhersagen (Blurriness)

### 📝 Prüfungsrelevante Fragen
1. Wie funktioniert GAN Training? Erklären Sie Generator und Discriminator!
2. Was ist Mode Collapse und wie wird es gelöst?
3. Was ist der Unterschied zwischen BCE und Wasserstein Loss?
4. Wie funktioniert der VAE Reparametrisierungs-Trick?
5. Was ist die ELBO und ihre Komponenten?
6. Erklären Sie Forward und Reverse Process bei Diffusionsmodellen!
7. Was ist Classifier-Free Guidance?
8. Was misst FID und wie funktioniert es?

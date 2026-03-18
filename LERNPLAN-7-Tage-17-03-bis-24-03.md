# 7-Tage-Lernplan: Advanced ML (17.03. - 24.03.2026)

**Klausurdatum:** 24.03.2026  
**Heute:** Mittwoch, 18.03.2026  
**Verbleibend:** 6 Tage

---

## TAG 1 - DIENSTAG 17.03.
**Fokus:** Transformers & Self-Attention (🔴 SEHR WICHTIG)  
**Ziel:** Grundkonzept von Self-Attention, Q/K/V, Multi-Head verstehen

### Morgens (2.5 Stunden)
- [ ] 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md lesen (Seiten 1-30)
- [ ] Self-Attention Formel verstehen: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- [ ] Encoder vs Decoder Architektur skizzieren
- [ ] Positional Encoding Zweck notieren (nicht Formel auswendig!)
- [ ] Multi-Head Attention Konzept verstehen
- [ ] 📺 Video: "Attention in transformers, visually explained" - 3Blue1Brown (20 Min)  
  https://www.youtube.com/watch?v=eMlx5fFNoYc
- [ ] 📺 Video: "Transformer Neural Networks - EXPLAINED!" - CodeEmporium (15 Min)  
  https://www.youtube.com/watch?v=TQQlZhbC5ps

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen beantworten:
  - Was ist Self-Attention? Erklären Sie Q, K, V!
  - Warum braucht man Positional Encoding?
  - Was ist der Unterschied zwischen Encoder und Decoder?
  - Wie funktioniert Multi-Head Attention?
  - Was ist Masked Attention?
- [ ] Encoder-Only vs Decoder-Only Unterschiede notieren

### Abends (1 Stunde)
- [ ] Kurze Wiederholung der 3 Kernpunkte
- [ ] Offene Fragen für morgen notieren
- [ ] 📺 Video: "BERT vs GPT" - AI Explained (10 Min)  
  https://www.youtube.com/watch?v=9Ou9ReO9cew

**Selbsttest:** Kannst du Self-Attention mit eigenen Worten erklären?

---

## TAG 2 - MITTWOCH 18.03. (HEUTE)
**Fokus:** LSTM & Word Embeddings (🔴 SEHR WICHTIG)  
**Ziel:** LSTM Gates, Word2Vec, Bahdanau-Attention verstehen

### Morgens (2.5 Stunden)
- [ ] 04_ZUSAMMENFASSUNG-02-WordEmbeddings-RNNs.md lesen
- [ ] LSTM 3 Gates zeichnen (Forget, Input, Output)
- [ ] Cell State Update nachvollziehen
- [ ] Word2Vec (CBOW vs Skip-gram) Unterschiede notieren
- [ ] Bahdanau-Attention für Seq2Seq verstehen
- [ ] 📺 Video: "Word Embedding and Word2Vec, Clearly Explained!!!" - StatQuest (17 Min)  
  https://www.youtube.com/watch?v=viZrOnJclY0
- [ ] 📺 Video: "LSTM - EXPLAINED!" - CodeEmporium (18 Min)  
  https://www.youtube.com/watch?v=QciIc0qJ6Xs

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Erklären Sie die 3 Gates im LSTM!
  - Was ist der Unterschied zwischen CBOW und Skip-gram?
  - Was ist die Distributional Hypothesis?
  - Wie funktioniert Bahdanau-Attention?
  - Vanishing Gradients Problem bei RNNs?
- [ ] 📺 Video: "Attention Mechanism in Neural Networks" - StatQuest (15 Min)  
  https://www.youtube.com/watch?v=PSs6nxcdjaQ

### Abends (1 Stunde)
- [ ] Transformer vs LSTM Vergleich schreiben
- [ ] Bidirektionale RNNs/LSTMs verstehen
- [ ] 📺 Video: "Seq2Seq Models - EXPLAINED!" - CodeEmporium (12 Min)  
  https://www.youtube.com/watch?v=DejHQYAGb7Q

**Selbsttest:** Kannst du LSTM-Gates aufzeichnen und erklären?

---

## TAG 3 - DONNERSTAG 19.03.
**Fokus:** Q-Learning, Double DQN, Policy Gradients & Actor-Critic (🔴 SEHR WICHTIG)  
**Ziel:** RL-Grundlagen verstehen, Policy-Methoden kennen

### Morgens (2.5 Stunden)
- [ ] 09_ZUSAMMENFASSUNG-07-RL-Teil-1.md lesen (Seiten 1-50)
- [ ] Q-Learning Update-Formel auswendig: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
- [ ] Bellman-Optimalitätsgleichung verstehen
- [ ] Double DQN: Overestimation Problem notieren
- [ ] Experience Replay & Target Networks Zweck
- [ ] Policy Gradients Grundkonzept (REINFORCE)
- [ ] Actor-Critic Architektur (Vorteil: lower variance)
- [ ] 📺 Video: "Bellman Equation - Explained!" - CodeEmporium (15 Min)  
  https://www.youtube.com/watch?v=9JZID-h6ZJ0
- [ ] 📺 Video: "Q-Learning Explained" - CodeEmporium (18 Min)  
  https://www.youtube.com/watch?v=__t2EVm7-wo

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Was ist der Unterschied zwischen Q-Learning und SARSA?
  - Was ist das Overestimation Problem?
  - Wie löst Double DQN das Problem?
  - Unterschied Value-Based vs Policy-Based Methods?
  - Was ist der Vorteil von Actor-Critic?
- [ ] 📺 Video: "Double DQN - EXPLAINED!" - CodeEmporium (10 Min)  
  https://www.youtube.com/watch?v=0cF4q-x6Q4Y

### Abends (1 Stunde)
- [ ] Kurze Formel-Wiederholung
- [ ] On-Policy vs Off-Policy Unterschiede
- [ ] 📺 Video: "Policy Gradient Methods - EXPLAINED!" - CodeEmporium (16 Min)  
  https://www.youtube.com/watch?v=5P7I-xPqHRU

**Selbsttest:** Kannst du Q-Learning Update ohne Nachschauen schreiben?

---

## TAG 4 - FREITAG 20.03.
**Fokus:** GANs & VAEs - VERTIEFT (🔴 WICHTIG)  
**Ziel:** Generator/Discriminator, Conditional GAN, Controllable Generation

### Morgens (2.5 Stunden)
- [ ] 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md lesen (Seiten 1-40)
- [ ] GAN Grundkonzept: Generator & Discriminator
- [ ] Minimax Loss verstehen
- [ ] Mode Collapse Problem notieren
- [ ] Conditional GAN (cGAN) - Konditionierung auf Labels
- [ ] Controllable Generation (Attribute manipulation)
- [ ] Fidelity vs Diversity Trade-off
- [ ] VAE: Reparametrisierungs-Trick, ELBO Loss
- [ ] 📺 Video: "Generative Adversarial Networks (GANs) - EXPLAINED!" - CodeEmporium (18 Min)  
  https://www.youtube.com/watch?v=8LlnWUXe1P8
- [ ] 📺 Video: "Variational Autoencoders (VAEs) - EXPLAINED!" - CodeEmporium (16 Min)  
  https://www.youtube.com/watch?v=fcvY9YGv9y4

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Was ist Mode Collapse bei GANs?
  - Erklären Sie Generator vs Discriminator!
  - Was unterscheidet Conditional GAN vom Standard-GAN?
  - Was ist der Fidelity vs Diversity Trade-off?
  - Was ist die ELBO beim VAE?
- [ ] 📺 Video: "Mode Collapse in GANs - EXPLAINED!" - CodeEmporium (10 Min)  
  https://www.youtube.com/watch?v=2lZTMD8u1eE

### Abends (1 Stunde)
- [ ] GAN vs VAE vs Conditional GAN Vergleich (Tabelle)
- [ ] Leichte Wiederholung der Loss-Funktionen
- [ ] 📺 Video: "VAE Reparameterization Trick - EXPLAINED!" - CodeEmporium (8 Min)  
  https://www.youtube.com/watch?v=9j8T2WNIx2Q

**Selbsttest:** Kannst du den VAE Reparametrisierungs-Trick erklären?

---

## TAG 5 - SAMSTAG 21.03.
**Fokus:** Diffusion, Latent Diffusion & XAI - VERTIEFT (🟡 WICHTIG)  
**Ziel:** Forward/Reverse Process, Latent Diffusion, Counterfactuals

### Morgens (2.5 Stunden)
- [ ] 07_ZUSAMMENFASSUNG-05-GANs-VAEs-Diffusion.md (Seiten 31-60)
- [ ] Diffusion: Forward Process (Noise hinzufügen)
- [ ] Reverse Process (Denoising)
- [ ] Training Loss Konzept
- [ ] Latent Diffusion (Stable Diffusion) - Warum im Latent Space?
- [ ] Classifier-Free Guidance
- [ ] 06_ZUSAMMENFASSUNG-04-XAI.md lesen (Seiten 1-30)
- [ ] LIME: lokale Approximation
- [ ] SHAP: Shapley Values Grundidee
- [ ] Counterfactual Explanations (Was müsste sich ändern?)
- [ ] 📺 Video: "Diffusion Models - EXPLAINED!" - CodeEmporium (20 Min)  
  https://www.youtube.com/watch?v=HoKDTa5jHVA
- [ ] 📺 Video: "Latent Diffusion Models - EXPLAINED!" - CodeEmporium (12 Min)  
  https://www.youtube.com/watch?v=0YLJNq9OY_I

### Nachmittags (1.5 Stunden)
- [ ] 5 Selbsttestfragen:
  - Erklären Sie Forward und Reverse Process bei Diffusion!
  - Was ist Latent Diffusion und warum ist es effizienter?
  - Was ist Classifier-Free Guidance?
  - Wie funktioniert LIME?
  - Was sind Counterfactual Explanations?
- [ ] 📺 Video: "LIME (Local Interpretable Model-agnostic Explanations) - EXPLAINED!" (15 Min)  
  https://www.youtube.com/watch?v=8D08Z_lA2Xw

### Abends (1 Stunde)
- [ ] Diffusion vs GAN vs VAE vs Latent Diffusion Vergleich
- [ ] XAI Methoden Liste durchgehen
- [ ] 📺 Video: "SHAP (SHapley Additive exPlanations) - EXPLAINED!" - CodeEmporium (18 Min)  
  https://www.youtube.com/watch?v=9haUOq4qRQc

**Selbsttest:** Kannst du den Diffusion-Prozess in 3 Sätzen erklären?

---

## TAG 6 - SONNTAG 22.03.
**Fokus:** RL Exploration, Imitation Learning, RLHF & DPO (🟡 WICHTIG)  
**Ziel:** UCB, Thompson Sampling, DAgger, RLHF/DPO verstehen

### Morgens (2.5 Stunden)
- [ ] 10_ZUSAMMENFASSUNG-08-RL-Teil-2.md lesen (Seiten 1-35)
- [ ] UCB Formel: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
- [ ] Thompson Sampling Konzept
- [ ] Exploration vs Exploitation
- [ ] Offline RL (Schwierigkeit, Lösungsansätze)
- [ ] 08_ZUSAMMENFASSUNG-06-IL.md lesen
- [ ] Behavioral Cloning
- [ ] Distributional Shift Problem
- [ ] DAgger Algorithmus
- [ ] 📺 Video: "Upper Confidence Bound (UCB) - EXPLAINED!" - CodeEmporium (12 Min)  
  https://www.youtube.com/watch?v=6fE7gviVh2Y
- [ ] 📺 Video: "Thompson Sampling - EXPLAINED!" - CodeEmporium (14 Min)  
  https://www.youtube.com/watch?v=8NSKZ4KeO4A

### Nachmittags (1.5 Stunden)
- [ ] 05_ZUSAMMENFASSUNG-03-Transformers-LLMs.md (Seiten zu RLHF/DPO)
- [ ] RLHF (Reward-Modell, PPO-Training)
- [ ] DPO (Direct Preference Optimization) - Unterschied zu RLHF
- [ ] Constitutional AI (Grundidee)
- [ ] 5 Selbsttestfragen:
  - Wie funktioniert UCB?
  - Was ist Thompson Sampling?
  - Was ist Distributional Shift beim Imitation Learning?
  - Was sind die Schritte bei RLHF?
  - Was ist der Unterschied zwischen RLHF und DPO?
- [ ] 📺 Video: "DAgger (Dataset Aggregation) - EXPLAINED!" - CodeEmporium (10 Min)  
  https://www.youtube.com/watch?v=gffZio5UmQc

### Abends (1 Stunde)
- [ ] On-Policy vs Off-Policy Wiederholung
- [ ] Reasoning in LLMs (Chain-of-Thought, Tree of Thoughts) - kurz überblicken
- [ ] RAG (Retrieval-Augmented Generation) - Grundkonzept
- [ ] 📺 Video: "RLHF (Reinforcement Learning from Human Feedback) - EXPLAINED!" (15 Min)  
  https://www.youtube.com/watch?v=2MBJou9X46o

**Selbsttest:** Kannst du UCB-Formel schreiben und erklären?

---

## TAG 7 - MONTAG 23.03.
**Fokus:** PUFFERTAG - Wiederholung & Schwachstellen  
**Ziel:** Alle Themen überblicken, offene Fragen klären

### Morgens (2 Stunden)
- [ ] ALLE Zusammenfassungen überfliegen (03-10)
- [ ] Eigene Notizen durchgehen
- [ ] Schwache Themen identifizieren
- [ ] 32 Selbsttest-Fragen aus ANTWORTEN-Alle-32-Klausurfragen.md durchgehen
- [ ] Offene Fragen klären

### Nachmittags (2 Stunden)
- [ ] Schwache Themen gezielt wiederholen
- [ ] Wichtige Formeln wiederholen:
  - Self-Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  - Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
  - UCB: a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]
- [ ] CHEAT-SHEET durchgehen

### Abends (1 Stunde)
- [ ] LEICHTER Überblick - nicht mehr neu lernen!
- [ ] Früh ins Bett gehen!
- [ ] Materialien für morgen bereitlegen

**Selbsttest:** Fühlst du dich sicher bei allen 🔴 Themen?

---

## TAG 8 - DIENSTAG 24.03. - KLAUSURTAG 📝

### Morgens (1 Stunde vor Klausur)
- [ ] Gutes Frühstück!
- [ ] NUR kurze Wiederholung:
  - Self-Attention Konzept (1 Minute)
  - LSTM Gates (1 Minute)
  - Q-Learning Formel (1 Minute)
  - GAN Grundprinzip (1 Minute)
- [ ] NICHTS NEUES MEHR!

### In der Klausur
- [ ] Erst einfache Fragen lösen
- [ ] Bei Unsicherheit: Markieren und später zurückkommen
- [ ] Zeitmanagement: Nicht zu lange an einer Frage hängen
- [ ] Konzepte erklären, nicht alles auswendig!

**Erfolg!** 🎯

---

## 📊 Priorisierungsübersicht

| Priorität | Themen | Tage |
|-----------|--------|------|
| 🔴 SEHR WICHTIG | Transformers, LSTM, Q-Learning, Word Embeddings | Tag 1-3 |
| 🔴 WICHTIG | GANs (inkl. Conditional), VAE, Diffusion, Latent Diffusion | Tag 4-5 |
| 🟡 WICHTIG | XAI (inkl. Counterfactuals), RL Exploration, IL, RLHF/DPO | Tag 5-6 |
| 🟢 GRUNDWISSEN | Policy Gradients, Actor-Critic, RAG, Reasoning | Tag 3, 6 |

---

## 📝 Wichtige Formeln (auswendig)

1. **Self-Attention:** Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. **Q-Learning:** Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
3. **UCB:** a = argmax_a [μ̂_a + √(2·ln(T)/N(a))]

---

## ⚠️ Nicht klausurrelevant (NICHT lernen!)

❌ Tokenisierung Details (BPE etc.)  
❌ Aktivierungsfunktionen Details  
❌ Positional Encodings Formeln  
❌ Metriken & Benchmarks  
❌ Layer-wise relevance propagation  
❌ Integrated Gradients  
❌ Wasserstein GAN  
❌ DDPG, TRPO, PPO, SAC (Details)  
❌ Implementierungstipps  

---

## 🆕 ÄNDERUNGEN IN DIESER VERSION

**Hinzugefügt:**
- Tag 1: Vollständiger Inhalt für Transformers & Self-Attention
- Alle Tage: Video-Links mit URLs zu allen empfohlenen Videos
- Tag 3: Policy Gradients & Actor-Critic
- Tag 4: Conditional GAN, Controllable Generation, Fidelity vs Diversity
- Tag 5: Latent Diffusion, Counterfactual Explanations
- Tag 6: RLHF, DPO, Constitutional AI, Reasoning, RAG

**Alle Themen aus dem Inhaltsverzeichnis sind jetzt vollständig abgedeckt.**

---

**Letzte Aktualisierung:** 18.03.2026, 08:50 Uhr  
**Status:** ✅ VOLLSTÄNDIG

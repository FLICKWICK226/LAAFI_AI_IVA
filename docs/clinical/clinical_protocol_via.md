# 🩺 Protocole Clinique d'Examen IVA (Inspection Visuelle à l'Acide Acétique)

> **Dispositif Médical Logiciel (SaMD) :** LAAFI_AI_IVA Engine v2.0  
> **Contexte Terrain :** Centres de Santé et de Promotion Sociale (CSPS) & Équipes Mobiles (Burkina Faso / LMIC)  
> **Recommandation Référente :** Guide OMS 2021 sur le dépistage et le traitement des lésions précancéreuses du col de l'utérus.

---

## 1. Déroulement Standard de la Procédure "Screen-and-Treat"

```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ 1. Pose du Spéculum     │ ──► │ 2. Application Acétique │ ──► │ 3. Capture Smartphone   │
│ Exposer le col utérin   │     │ Acide Acétique 3% à 5%  │     │ 60 secondes post-badigeon│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ 6. Traitement / Réf     │ ◄── │ 5. Décision Triage LAAFI│ ◄── │ 4. Inférence Offline    │
│ Thermocoagulation / CHU │     │ Vert / Jaune / Rouge    │     │ CADe (YOLO) + CADx (AI) │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 2. Étapes de l'Examen

1. **Préparation & Pose du Spéculum :**
   - Mise en place du spéculum bivalve sans lubrifiant susceptible d'altérer la réactivité acétique.
   - Nettoyage doux du mucus excessif avec un coton sec si nécessaire.
2. **Badigeonnage à l'Acide Acétique (3% à 5%) :**
   - Imprégnation généreuse du col avec une compresse imbibée.
   - **Chronométrage obligatoire :** Attendre exactement **60 secondes** pour permettre la coagulation réversible des protéines cellulaires intracellulaires anormales.
3. **Capture Photographique avec Smartphone :**
   - Distance de prise de vue : 10 à 15 cm.
   - Flash LED activé pour éclairer uniformément le fond vaginal.
   - Vérification visuelle que le col est centré et net.
4. **Analyse Assistée par IA (LAAFI_AI_IVA) :**
   - L'infirmier ou la sage-femme charge l'image dans l'application Android LAAFI_AI (100% hors-ligne).
   - Stage 1 (CADe) isole automatiquement la Zone de Jonction Squamo-Columnaire (JSC).
   - Stage 2 (CADx) calcule la probabilité de lésion acéto-blanche précancéreuse (CIN2+).
5. **Prise en Charge Clinique Immédiate :**
   - **Vert ($P < 0.20$) :** Négatif. Contrôle de routine à 3 ans.
   - **Jaune ($0.20 \le P < 0.38$) :** Zone grise / Douteux. Réalisation d'un 2nd badigeon ou examen par un collègue senior.
   - **Rouge ($P \ge 0.38$) :** Positif. Si éligible, traitement immédiat par **thermocoagulation (ablation thermique)** sur place au CSPS ; sinon, référence vers le Centre Médical avec Antenne Chirurgicale (CMA) ou CHU.

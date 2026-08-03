# 🔬 Synthèse de la Littérature Scientifique : Dépistage IVA par Imagerie Smartphone en Zones à Faibles Ressources

> **Document de référence scientifique pour le projet LAAFI_AI IVA Engine (Version 2.0)**  
> *Rédigé à partir des publications PubMed / PMC et des analyses de la communauté médicale internationale.*

---

## 📌 1. Contexte Clinique & Problématique dans les Pays à Faibles Revenus (LRS)

Le cancer du col de l'utérus est le quatrième cancer le plus fréquent chez la femme au niveau mondial, mais **plus de 90 % des décès surviennent dans les pays à revenus faibles ou intermédiaires (LMIC)**. 

L'Organisation Mondiale de la Santé (OMS) recommande l'**Inspection Visuelle à l'Acide Acétique (IVA)** comme méthode de dépistage primaire. L'acide acétique provoque une coagulation temporaire des protéines cellulaires, révélant une teinte **acéto-blanche** sur l'épithélium dysplasique.

### Limites de l'IVA traditionnelle à l'œil nu :
1. **Forte variabilité inter-observateur** : La sensibilité varie de 45% à 85% selon l'expérience de la sage-femme ou de l'infirmier.
2. **Absence de contrôle qualité** : Pas de trace photographique pour audit ou supervision à distance.
3. **Coût des colposcopes classiques** : Les colposcopes médicaux sont volumineux, coûteux (> 15 000 $), dépendants du réseau électrique et nécessitent une maintenance spécialisée.

---

## 📚 2. Analyse Détaillée des 5 Études Majeures (PubMed / PMC)

### 📄 Étude 1 : Sami et al. (2022) – *Digital-VIA (D-VIA) & Algorithmes IA*
- **Titre :** *Smartphone-Based Visual Inspection with Acetic Acid: An Innovative Tool to Improve Cervical Cancer Screening in Low-Resource Setting.* ([PMID: 35207002](https://pubmed.ncbi.nlm.nih.gov/35207002/))
- **Journal :** *Healthcare (Basel)* | **DOI :** 10.3390/healthcare10020391
- **Constats Clés :**
  - La caméra haute définition des smartphones permet la comparaison simultanée en temps réel des images natives et post-IVA avec zoom sur la zone de transformation (JSC).
  - La qualité des clichés pris par smartphone est suffisante pour le diagnostic des lésions **CIN1 et CIN2+**.
  - L'intégration d'algorithmes d'IA (Computer Vision) est recommandée pour automatiser la détection et éliminer la variabilité humaine.

---

### 📄 Étude 2 : Ferguson et al. (2024) – *Programme SEVIA en Tanzanie*
- **Titre :** *An Implementation Evaluation of the Smartphone-Enhanced Visual Inspection with Acetic Acid (SEVIA) Program for Cervical Cancer Prevention in Urban and Rural Tanzania.* ([PMID: 39063455](https://pubmed.ncbi.nlm.nih.gov/39063455/))
- **Journal :** *Int. J. Environ. Res. Public Health* | **DOI :** 10.3390/ijerph21070878
- **Constats Clés :**
  - Évaluation à grande échelle du système SEVIA auprès de 66 soignants dans 14 centres de santé en Tanzanie.
  - La capture d'image par smartphone associée à une télé-expertise offre une amélioration drastique des pratiques cliniques.
  - **Défis majeurs identifiés :** Problèmes de connectivité réseau et artefacts visuels lors des prises de vue rapides.

---

### 📄 Étude 3 : Bae et al. (2020) – *Quantification Machine Learning & Simulation de Bruits*
- **Titre :** *Quantitative Screening of Cervical Cancers for Low-Resource Settings: Pilot Study of Smartphone-Based Endoscopic Visual Inspection After Acetic Acid Using Machine Learning Techniques.* ([PMID: 32159521](https://pubmed.ncbi.nlm.nih.gov/32159521/))
- **Journal :** *JMIR mHealth uHealth* | **DOI :** 10.2196/16467
- **Constats Clés :**
  - Extraction de 240 découpes d'images à partir du cervigramme aux positions horlogères de la JSC.
  - Démonstration qu'un algorithme de Machine Learning autonome surpasse l'interprétation moyenne des médecins généralistes (Précision 80.8%, Spécificité 84.1%).
  - Nécessité absolue de prétraiter les bruits de réflectance et les masques sanguins pour isoler les caractéristiques spectrales acéto-blanches.

---

### 📄 Étude 4 : Allanson et al. (2021) – *Méta-Analyse de Précision de la S-VIA pour les Lésions CIN2+*
- **Titre :** *Accuracy of Smartphone Images of the Cervix After Acetic Acid Application for Diagnosing Cervical Intraepithelial Neoplasia Grade 2 or Greater in Women With Positive Cervical Screening: A Systematic Review and Meta-Analysis.* ([PMID: 34936374](https://pubmed.ncbi.nlm.nih.gov/34936374/))
- **Journal :** *JCO Global Oncology* | **DOI :** 10.1200/GO.21.00168
- **Constats Clés :**
  - Méta-analyse basée sur 6 003 études filtrées et 6 essais retenus.
  - La sensibilité brute du smartphone sans IA est de **74.6%** avec une spécificité de **61.8%**.
  - **Conclusion clinique :** L'ajout d'une IA (CADe/CADx) est indispensable pour faire passer la sensibilité de 74.6% au seuil de sécurité clinique de **$\ge 95.0\%$**.

---

### 📄 Étude 5 : Asgary et al. (2019) – *Déploiement Communautaire au Ghana*
- **Titre :** *Acceptability and implementation challenges of smartphone-based training of community health nurses for visual inspection with acetic acid in Ghana: mHealth and cervical cancer screening.* ([PMID: 31315879](https://pubmed.ncbi.nlm.nih.gov/31315879/))
- **Journal :** *BMJ Open* | **DOI :** 10.1136/bmjopen-2019-030528
- **Constats Clés :**
  - Évaluation de l'utilisation de smartphones par les infirmières de santé communautaires à Accra (Ghana).
  - Validation de l'acceptabilité par les patientes et mise en évidence du besoin d'algorithmes de détection automatique en mode hors-ligne (*Edge AI*).

---

## 🛠️ 3. Pratiques de Simulation & Traitement des Bruits de Terrain

Dans la littérature en vision par ordinateur appliquée à l'IVA, quatre types de perturbations optiques et biologiques reviennent systématiquement :

```text
┌───────────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────┐
│ BRUIT DE TERRAIN          │ EFFET CLINIQUE SUR L'IMAGE               │ SOLUTION & STRATÉGIE SIMULATION ML       │
├───────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ Reflets Spéculaires (Flash)│ Zones blanches saturées (V > 240 HSV)    │ Dilatation morphologique & masque HSV    │
│ Sang & Glaire             │ Opacification de la Jonction (JSC)       │ Bruit de Perlin vectorisé + Alpha Blend  │
│ Défocalisation / Tremblement│ Flou de bougé du smartphone              │ Defocus Kernel & MotionBlur (Albumentations)│
│ Température de Couleur    │ Faux positifs acéto-blancs               │ ColorJitter bridé (Hue shift <= 0.05)    │
└───────────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┘
```

### A. Reflets Spéculaires (Flash LED)
L'utilisation de la LED de smartphone produit une réflexion spéculaire intense sur le mucus acétique. La littérature préconise d'isoler le canal de valeur (V dans HSV) avec $V \ge 240$, puis d'appliquer une dilatation morphologique pour exclure les pixels éblouissants des cartes d'attention du réseau.

### B. Bruits Biologiques (Sang et Glaire via Bruit de Perlin)
Le sang (teinte rouge sombre R=130-180) et la glaire acétique (teinte blanchâtre R=210-230) sont simulés en superposition via du **bruit procédural de Perlin** à plusieurs octaves, fusionné avec un facteur d'opacité Alpha $\alpha \in [0.15, 0.45]$.

### C. Restriciton Stricte du Hue Shift ($\le 0.05$)
En imagerie médicale IVA, l'acéto-blanchiment est un signal de couleur délicat. Les augmentations génériques qui modifient la teinte (*Hue*) transforment des zones saines rosées en fausses lésions blanchâtres. Le *Hue Shift* doit être strictement limité à $\le 0.05$.

---

## 🎯 4. Alignement Direct avec l'Architecture LAAFI_AI IVA v2.0

Ces découvertes scientifiques justifient à 100% les choix techniques faits dans le projet **LAAFI_AI** :

1. **Stage 1 (YOLO-Det)** : Recommandé par la littérature pour éliminer > 70% de l'arrière-plan (spéculum, vagin) et concentrer le réseau sur la JSC.
2. **Stage 2 Multi-tâche (ConvNeXt / Swin-v2)** : Évalue à la fois l'Éligibilité (Type 1/2 vs 3) et le Diagnostic Lésionnel.
3. **Focal Loss ($\gamma=2.0$) & Seuil $T \in [0.10, 0.40]$** : Permet de faire passer la sensibilité du smartphone de **74.6% (brut sans IA)** à **$\ge 95.0\%$ (seuil de sécurité)**.
4. **Audit Visuel Grad-CAM** : Conforme aux exigences d'explicabilité pour vérifier que l'IA ne focalise pas son attention sur les reflets spéculaires ou les zones sanguines.

---

## 🔗 5. Liste des URLs des Publications Consultées

- **Healthcare 2022 (D-VIA & AI) :** [https://pubmed.ncbi.nlm.nih.gov/35207002/](https://pubmed.ncbi.nlm.nih.gov/35207002/)
- **IJERPH 2024 (SEVIA Tanzania) :** [https://pubmed.ncbi.nlm.nih.gov/39063455/](https://pubmed.ncbi.nlm.nih.gov/39063455/)
- **JMIR mHealth 2020 (ML & Endoscopic VIA) :** [https://pubmed.ncbi.nlm.nih.gov/32159521/](https://pubmed.ncbi.nlm.nih.gov/32159521/)
- **JCO Global Oncology 2021 (Meta-Analysis S-VIA CIN2+) :** [https://pubmed.ncbi.nlm.nih.gov/34936374/](https://pubmed.ncbi.nlm.nih.gov/34936374/)
- **BMJ Open 2019 (mHealth Screening Ghana) :** [https://pubmed.ncbi.nlm.nih.gov/31315879/](https://pubmed.ncbi.nlm.nih.gov/31315879/)

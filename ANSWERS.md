  ## 🔴 Incohérences Majeures (Les "Showstoppers")

  ### 1. Incohérence des Labels : Ton Dataset ne contient pas ce que tu veux prédire

  • Ton objectif : Classifier les lésions en Négatif, Positif, Invalide (CADx).
  • La réalité du dataset : Le dataset Kaggle intel-mobileodt-cervical-cancer-screening contient
  uniquement des dossiers Type_1, Type_2, Type_3. Ces types correspondent à la visibilité anatomique
  de la Jonction Squamo-Columnaire (Type 1 : entièrement visible ; Type 2 : partiellement visible ;
  Type 3 : invisible/endocervicale).
  • Le diagnostic : C'est une erreur critique. Tu ne peux pas entraîner un classifieur de lésion
  (sain/cancer) sur un dataset étiqueté uniquement avec des types anatomiques. Ton modèle apprendra
  à prédire la forme du col de l'utérus, pas la présence de cellules cancéreuses. Pour classifier
  les lésions, il te faut des labels pathologiques (ex: CIN 1/2/3, bénin/malin), absents de ce
  dataset sous cette forme.

  ### 2. YOLO Segmentation sans annotations de segmentation

  • Ton objectif : Entraîner YOLOv8-Seg ou YOLOv11-Seg (CADe) pour détecter et segmenter la JSC.
  • La réalité du dataset : Le dataset brut d'Intel MobileODT ne contient aucune image segmentée au
  pixel près (masks) pour la JSC. Il contient des images brutes classées par répertoires.
  • Le diagnostic : Comment veux-tu entraîner un modèle de segmentation sans masques de segmentation
  ? Soit tu dois annoter manuellement des milliers d'images, soit tu dois rétrograder ton Stage 1 à
  de la détection d'objets (Bounding Boxes) en récupérant des annotations de la communauté Kaggle,
  soit changer de dataset.

  ### 3. Le goulot d'étranglement CPU de Perlin Noise

  • Ton implémentation : Les lignes LAAFI_AI_IVA_PRD.md génèrent le bruit de Perlin pixel par pixel
  avec une double boucle Python for i in range(h): for j in range(w):.
  • La réalité : Pour une image 384 × 384, cela fait 147 456 itérations par image. Pour un batch de
  32, c'est 4,7 millions d'itérations CPU par batch.
  • Le diagnostic : Ton code d'augmentation est inutilisable en conditions réelles. Ton CPU va
  saturer à 100%, ton GPU Tesla T4 de Colab restera à 0% d'utilisation à attendre les données. Tu
  dois vectoriser cette opération via NumPy ou PyTorch ou générer les masques de bruit hors-ligne
  avant l'entraînement.

  ### 4. Patient-Level Split impossible avec les métadonnées de base

  • Ta règle : Pas de fuite de données (Rule 1), split strict par Patient_ID.
  • La réalité du dataset : Les images Kaggle sont nommées de façon séquentielle (1.jpg, 2.jpg...).
  Il n'y a pas de fichier de métadonnées officiel liant chaque image à un Patient_ID.
  • Le diagnostic : Sans cartographie image -> Patient_ID (que tu dois reconstruire via du
  clustering visuel ou des scripts communautaires), tu feras un split aléatoire par image, violant
  directement ta propre règle anti-data-leakage.
  ──────
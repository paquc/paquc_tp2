# Prédiction d'alarmes par AI

## Structure du repo GIT
Le lien du repo est `https://github.com/paquc/paquc_tp2`

La structure est la suivante:
- Root
	- Scripts
		- LogParsing: tous les scripts python ou autres servant à la préparation des logs, séquences et matrices d'occurences
		
## LogParser Docker Container


## Préparation des logs

1. Génération des logs structurés et templates

Pour générer le log structuré et les templates de BGL, il faut exécuter le script `Root/Scripts/LogParsing/BGLBrainParse.py`.
Ce qui générera les fichiers `BGL.log_structured.csv` et `BGL.log_templates.csv`.

???? 2. Fragmentation du log structuré

Problème de génération averc un log trop gros comme BGL. Utilisation de la fragmentation du log. PAS IDÉAL!!!!!

Pour splitter, exécuter `Root/Scripts/LogParsing/BGLLogFileSplit.py`

3. Générations des séquences, matrices d'occurences et élimination de diplicatas

Nous avons choisi de générer des matrices d'occurences pour la détection des ALARMES.
Génération de séquences d'exécution dans le temps avec une fenêtre de 100 événement précédant une alarme pour la détection d'alarmes.
L'alarme KERNDTLB est la plus fréquente, elle a donc été utilisée comme alarme principale.
Le concept s'applique à tous les autres types d'alarmes.

??? Exécuter `Root/Scripts/LogParsing/BGLGenOccurencesMatricesV1.py` pour générer les fichiers `KERNDTLB_alarm_occurences_matrix_V1_part<partno>_dedup.csv`

Exécuter `Root/Scripts/LogParsing/BGLGenOccurencesMatricesV3.py` pour générer les fichiers `KERNDTLB_alarm_occurences_matrix_V3_dedup.csv`

4. Entreinement des modèles et génération des métriques

Pour entreiner les modèles pour BGL, exécuter `Root/Scripts/ModelsTraining/BGL_Train_KERNDTLB_Alarms_RF_BootStrap.py`.
Le fichier `BGL_KERNDTLB_Trainig_Output.log` est généré avec les métriques et la classification.


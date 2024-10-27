# Prédiction d'alarmes par AI

## Structure du repo GIT
Le lien du repo est `https://github.com/paquc/paquc_tp2`

La structure est la suivante:
- Root
	- Scripts
		- LogParsing: tous les scripts python ou autres servant à la préparation des logs, séquences et matrices d'occurences
		
## LogParser Docker Container


## Prédictions d'alarmes pour BGL

1. Génération des logs structurés et templates

BGL.log contient 4,747,964 lignes.

Pour générer le log structuré et les templates de BGL, il faut exécuter le script `Root/Scripts/LogParsing/BGLBrainParseV4.py`.
Ce script prend en entrée le fichier BGL.log qui contient plus de 4,747,963 d'événements.
Ce qui générera les fichiers `BGL.log_structured_V4.csv` et `BGL.log_templates_V4.csv`.

2. Générations des séquences, matrices d'occurences et élimination de diplicatas

Nous avons choisi de générer des matrices d'occurences pour la détection des ALARMES.
Génération de séquences d'exécution dans le temps avec une fenêtre de 100 événement précédant une alarme pour la détection d'alarmes.
L'alarme KERNDTLB est la plus fréquente, elle a donc été utilisée comme alarme principale.
Le concept s'applique à tous les autres types d'alarmes.

Exécuter `Root/Scripts/LogParsing/BGLGenOccurencesMatricesV4.py` pour générer les fichiers `KERNDTLB_alarm_occurences_matrix_V4_dedup.csv`

3. Entreinement des modèles et génération des métriques

Pour entreiner les modèles pour BGL, exécuter `Root/Scripts/ModelsTraining/BGL_Train_KERNDTLB_Alarms_RF_LR_BootStrap_Full_Param.py V4`.
Le fichier `BGL_KERNDTLB_Training_V4_Output.log` est généré avec les métriques et la classification.

## Prédictions d'alarmes pour Thunderbird

1. Génération des logs structurés et templates

Un log de 5,000,000 de lignes a été utilisé afin de comparer correctement avec BGL.

Pour générer le log structuré et les templates, il faut exécuter le script `Root/Scripts/LogParsing/ThuBrainParseV4.py`.
Ce script prend en entrée le fichier Thunderbird_5M.log qui contient plus de 5,000,000 d'événements.
Ce qui générera les fichiers `Thunderbird_5M.log_structured.csv` et `Thunderbird_5M.log_templates.csv`.

2. Générations des séquences, matrices d'occurences et élimination de diplicatas

Nous avons choisi de générer des matrices d'occurences pour la détection des ALARMES.
Génération de séquences d'exécution dans le temps avec une fenêtre de 100 événement précédant une alarme pour la détection d'alarmes.
L'alarme VAPI (226071 instances) est la plus fréquente, elle a donc été utilisée comme alarme principale.
Le concept s'applique à tous les autres types d'alarmes.

Exécuter `Root/Scripts/LogParsing/ThuGenOccurencesMatricesV4.py` pour générer les fichiers `VAPI_alarm_occurences_matrix_V4_dedup.csv`

3. Entreinement des modèles et génération des métriques

Pour entreiner les modèles pour Thunderbird, exécuter `Root/Scripts/ModelsTraining/Thu_Train_VAPI_Alarms_RF_LR_BootStrap_Full_Param V4`.
Le fichier `Thu_VAPI_Training_V4_Output.log` est généré avec les métriques et la classification.






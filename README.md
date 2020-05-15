# thepokemonrecogniszeerdotio

#### Gruppemedlemmer

* Karl Frödin - cph-kf112
* Emil Gotthelf Tranberg Hansen - eh130
* Jesper Holm Andersen - ja284
* Ulrik Holm - uh76
* Jeppe Juul - jj443

#### Projektbeskrivelse

Exam project proposal

Vi vil holde os inden for de første 151 pokemoner (gen 1)

Tech stack:
* Keras - machine learning
* Flask - webapp
* Matplotlib - diagrammer
* pymysql - database


1. I den første del vil vi træne en machine learning model til at genkende Pokémon 
2. I anden del vil vi lave et program hvor man giver et billede af en pokémon, hvor det via neural network kan genkende den.
3. I tredje del vil programmet hente information om pokémonen, der ligger i en database og vise det på en html side.

#### Dataset

dataset = 'https://mega.nz/file/7YtkUaCB#3HaajMQPlxiYeCemgb5-WRJNpIM1ENt3TV4oFXUZu-c'

#### Brugsvejledning: 

1. Opdater informationer i settings.py
2. Opret en 'input' mappe der passer med din UPLOAD_PATH i settings
2. Lav en database ved navn 'pokemon' og kør dump.sql filen
3. Installer nødvendige pakker:

    `conda install keras flask pymysql opencv-python`
    
4. Fra roden af projektet kør denne kommando i terminalen:

    `python app.py`
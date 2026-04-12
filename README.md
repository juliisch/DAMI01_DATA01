# DAMI01_DATA01
Dieses Repository enthält den Code-Teil der Studienarbeit für das Modul DAMI01 / DATA01 Data Analytics. 

In diesem Projekt wird eine Time Series Clustering (TSC) auf den Transaktionsdatensatz [Financial Transactions Dataset: Analytics von Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) angewendet. 

Unter der Datei **[Variablenbeschreibung](DATA_INFORMATION.md)** ist eine Beschreibung der in den Datensätzen vorkommenden Variablen.

## Projektstruktur

```
data_analytics_master/
├── README.md                                       # Diese Datei
├── DATA_INFORMATION.md                             # Variablenbeschreibung der verschiedenen Datensätze
├── requirements.txt                                # Python-Anforderungen (zu installierende Bibliotheken)
├── data/                                           # Daten
│   ├── origin/                                     # Originaldaten (gitignored)
│   └── processed                                   # Verarbeitete Daten 
├── notebooks/                                      # Datensammlung
│   ├── 01_Business_Data_Understanding.ipynb/       # Geschäftsverständnis und EDA
│   ├── 02_Data_Preparation.ipynb/                  # Datenaufbereitung
│   ├── 03_Modeling_Evaluation.ipynb                # Clustering und Evaluation
│   ├── funktionen.py                               # Definierte Funktionen
│   └── parameter.py                                # Globale Parameter und Map-Dictionaries
└── output/                                         # Generierten Grafiken und ....

```

### Installation und Ausführung der Simulation

1. **Klonen Sie das Repository und wechseln Sie in das Verzeichnis**

    ```bash
    git clone git@github.com:juliisch/DAMI01_DATA01.git
    ```
    ```bash
    cd DAMI01_DATA01
    ```


2. **Setze eine virtuelle Umgebung auf**
    ```bash
    python3 -m venv dami01_env
    ```
    ```bash
    source dami01_env/bin/activate
    ```
    ```bash
    pip install ipykernel
    ```

    ```bash
    python -m ipykernel install --user --name=dami01_env --display-name "Python (dami01_env)"
    ```

3. **Bibliotheken installieren**

    Installieren Sie die benötigten Bibliotheken über die Datei `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    Nach der Installation der Bibliotheken kann es vorkommen, dass ein Neustart des Programms erforderlich ist, damit die Bibliotheken wirksam werden.

4. **Führen Sie die Notebooks nacheinander aus**

    - `notebooks/01_Business_Data_Understanding.iypnb`
    - `notebooks/02_Data_Preparation.iypnb`
    - `notebooks/03_Modeling_Evaluation.iypnb`
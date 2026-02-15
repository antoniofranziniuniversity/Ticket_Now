# 🎫 Sistema Intelligente Smistamento Ticket - MVP

Questo progetto è un Minimum Viable Product (MVP) sviluppato per automatizzare lo smistamento dei ticket aziendali. 
A partire dal testo del ticket (Oggetto + Descrizione), il sistema utilizza algoritmi di Machine Learning (TF-IDF + Regressione Logistica) per prevedere automaticamente:
1. La **Categoria** di competenza (Amministrazione, Tecnico, Commerciale).
2. La **Priorità** operativa (Bassa, Media, Alta).

## 📁 Struttura del Progetto

Il repository è organizzato come segue:

* `data/`: Contiene il dataset sintetico generato per l'addestramento.
* `models/`: Contiene i modelli di ML addestrati (salvati in formato .pkl).
* `reports/`: Contiene gli output visivi della valutazione (es. Matrici di Confusione).
* `src/`: Contiene il codice sorgente Python diviso in tre script.
* `requirements.txt`: Elenco delle librerie necessarie per eseguire il progetto.

## ⚙️ Installazione e Prerequisiti

Assicurati di avere Python installato. Si consiglia di utilizzare un ambiente virtuale (es. venv su PyCharm).
Per installare tutte le dipendenze necessarie, apri il terminale nella root del progetto ed esegui:

    pip install -r requirements.txt


## 🚀 Come eseguire il progetto (Step-by-Step)

Il progetto è diviso in tre fasi sequenziali. Esegui i comandi dal terminale, assicurandoti di essere nella cartella principale del progetto.

### Fase 1: Generazione del Dataset Sintetico
Crea un dataset di ticket simulati (200-500 righe) basato su parole chiave e regole di priorità.

    python src/1_generate_dataset.py

Output atteso: Creazione del file `synthetic_tickets.csv` nella cartella `data/`.

### Fase 2: Addestramento e Valutazione Modelli (Pipeline ML)
Legge il CSV, esegue il preprocessing, vettorizza il testo (TF-IDF) e addestra due modelli di Regressione Logistica.

    python src/2_train_pipeline.py

Output atteso: Metriche (Accuracy, F1) a schermo. Salvataggio dei modelli in `models/` e dei grafici in `reports/`.

### Fase 3: Avvio della Dashboard Interattiva
Lancia l'interfaccia web (Streamlit) per testare il modello con ticket singoli o tramite caricamento massivo (CSV).

    streamlit run src/3_app.py

Output atteso: Si aprirà automaticamente una pagina nel browser all'indirizzo http://localhost:8501.

## 🛠️ Tecnologie Utilizzate
* **Linguaggio:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-learn (TfidfVectorizer, LogisticRegression)
* **Visualizzazione:** Matplotlib
* **Web UI:** Streamlit
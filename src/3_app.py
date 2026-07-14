import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import io

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Ticket Classifier MVP", page_icon="🎫", layout="centered")
st.title("🎫 Ticket Now")
st.markdown("Prototipo MVP per la classificazione automatica di categoria e priorità.")


# --- 2. GESTIONE PERCORSI E CARICAMENTO MODELLI ---
@st.cache_resource  # Evita di ricaricare i modelli a ogni interazione
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, "models")

    try:
        vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
        clf_cat = joblib.load(os.path.join(models_dir, "model_category.pkl"))
        clf_prio = joblib.load(os.path.join(models_dir, "model_priority.pkl"))
        return vectorizer, clf_cat, clf_prio
    except FileNotFoundError:
        st.error("❌ Modelli non trovati! Assicurati di aver eseguito '2_train_pipeline.py' prima.")
        st.stop()


vectorizer, clf_cat, clf_prio = load_models()


# Funzione per estrarre le parole più influenti
def get_top_words(text, vectorizer, model, prediction, top_n=5):
    # Trasforma il testo e ottieni i nomi delle feature (parole)
    vec_text = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Trova l'indice della classe predetta
    class_idx = np.where(model.classes_ == prediction)[0][0]

    # Moltiplica i coefficienti del modello per la presenza della parola nel testo
    # (Prendiamo i coefficienti specifici per la classe predetta)
    if len(model.classes_) == 2:  # Caso binario (non è il nostro caso, ma per sicurezza)
        coef = model.coef_[0] if class_idx == 1 else -model.coef_[0]
    else:
        coef = model.coef_[class_idx]

    scores = vec_text.toarray()[0] * coef

    # Ordina gli indici in base allo score decrescente
    top_indices = scores.argsort()[-top_n:][::-1]

    # Filtra solo le parole che hanno uno score maggiore di 0 (hanno contribuito positivamente)
    top_words = [feature_names[i] for i in top_indices if scores[i] > 0]
    return top_words


# --- 3. INTERFACCIA UTENTE (TABS) ---
tab1, tab2 = st.tabs(["📝 Ticket Singolo", "📂 Elaborazione Batch (CSV)"])

# TAB 1: TICKET SINGOLO
with tab1:
    st.header("Analizza un nuovo ticket")

    # Input utente
    ticket_title = st.text_input("Oggetto del ticket:", placeholder="Es. Problema con fattura")
    ticket_body = st.text_area("Descrizione:", placeholder="Es. Buongiorno, non riesco a scaricare la fattura...")

    if st.button("Analizza Ticket", type="primary"):
        if ticket_title and ticket_body:
            full_text = ticket_title + " " + ticket_body

            # Predizioni
            text_vec = vectorizer.transform([full_text])
            pred_cat = clf_cat.predict(text_vec)[0]
            pred_prio = clf_prio.predict(text_vec)[0]

            # Estrazione parole influenti
            top_words_cat = get_top_words(full_text, vectorizer, clf_cat, pred_cat)

            # UI dei risultati
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="🏢 Categoria Suggerita", value=pred_cat)
            with col2:
                # Diamo un colore alla priorità
                color = "red" if pred_prio == "Alta" else "orange" if pred_prio == "Media" else "green"
                st.markdown(f"### ⚡ Priorità: <span style='color:{color}'>{pred_prio}</span>", unsafe_allow_html=True)

            st.info(
                f"**Parole chiave che hanno influenzato la scelta della categoria:** {', '.join(top_words_cat) if top_words_cat else 'Nessuna di forte impatto'}")
        else:
            st.warning("Per favore, compila sia l'oggetto che la descrizione.")

# TAB 2: ELABORAZIONE BATCH
with tab2:
    st.header("Carica CSV per elaborazione massiva")
    st.markdown("Il CSV deve contenere le colonne `title` e `body`.")

    uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)

            if 'title' in df_batch.columns and 'body' in df_batch.columns:

                # 1. Mostra la rotellina di caricamento durante l'elaborazione
                with st.spinner("File caricato! Elaborazione in corso..."):

                    # Preprocessing
                    df_batch['full_text'] = df_batch['title'] + " " + df_batch['body']
                    batch_vec = vectorizer.transform(df_batch['full_text'])

                    # Predizioni
                    df_batch['predicted_category'] = clf_cat.predict(batch_vec)
                    df_batch['predicted_priority'] = clf_prio.predict(batch_vec)

                    # Rimuoviamo la colonna full_text di appoggio
                    df_batch = df_batch.drop(columns=['full_text'])

                # 2. Quando il blocco "with st.spinner" finisce, mostra il successo definitivo!
                st.success("✅ Elaborazione completata!")

                st.dataframe(df_batch.head(10))  # Mostra un'anteprima

                # Bottone per il download
                output = io.BytesIO()

                # Creiamo il file Excel
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_batch.to_excel(writer, index=False, sheet_name='Ticket_Predetti')

                    # Prendiamo il foglio di lavoro
                    worksheet = writer.sheets['Ticket_Predetti']

                    # Cicliamo su ogni colonna per calcolare la larghezza massima necessaria
                    for i, col in enumerate(df_batch.columns):
                        # Troviamo la lunghezza massima tra il nome della colonna e il contenuto delle celle
                        column_len = max(df_batch[col].astype(str).map(len).max(), len(col))
                        # Impostiamo la larghezza della colonna (aggiungiamo +2 di margine)
                        worksheet.set_column(i, i, column_len + 2)

                # Prepariamo il bottone per scaricare il file .xlsx
                st.download_button(
                    label="📥 Scarica Excel formattato",
                    data=output.getvalue(),
                    file_name="ticket_elaborati.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.error("Il file CSV deve contenere le colonne 'title' e 'body'.")
        except Exception as e:
            st.error(f"Si è verificato un errore nella lettura del file: {e}")

import pandas as pd
import random
import os

# Impostiamo un seed per garantire la riproducibilità (genererà sempre gli stessi dati)
random.seed(42)

# 1. Definiamo i vocabolari e i template per le 3 categorie aziendali
data_templates = {
    "Amministrazione": {
        "titles": ["Problema con fattura", "Richiesta IBAN", "Scadenza pagamento", "Errore in nota di credito",
                   "Bonifico non ricevuto"],
        "bodies": [
            "Buongiorno, non trovo la vostra fattura del mese scorso. Potete rimandarla?",
            "Necessito del vostro IBAN per effettuare il saldo in scadenza.",
            "Il pagamento risulta rifiutato dal sistema, come posso procedere?",
            "Ho notato un'incongruenza nella nota di credito emessa ieri.",
            "Abbiamo effettuato il bonifico ma non vedo lo sblocco amministrativo nel portale."
        ]
    },
    "Tecnico": {
        "titles": ["Errore di sistema", "Server non raggiungibile", "Reset password", "Software lento",
                   "Crash applicazione"],
        "bodies": [
            "Continuo a ricevere un errore 500 quando provo a salvare il form dei clienti.",
            "Il server principale non risponde, non riusciamo ad accedere ai file condivisi.",
            "Ho dimenticato la password e il link di reset non funziona, mi dà errore.",
            "Il gestionale è incredibilmente lento da questa mattina, è impossibile lavorare.",
            "L'applicazione va in crash improvvisamente appena clicco sul pulsante esporta."
        ]
    },
    "Commerciale": {
        "titles": ["Richiesta preventivo", "Informazioni su licenze", "Sconto per rinnovo", "Nuovo ordine",
                   "Dettagli offerta"],
        "bodies": [
            "Vorrei ricevere un preventivo dettagliato per 10 nuove utenze aziendali.",
            "Vorrei capire come funziona il passaggio al piano superiore di licenze software.",
            "Siamo in scadenza, è possibile avere uno sconto per il rinnovo annuale del contratto?",
            "Vorrei procedere con un nuovo ordine per il prodotto che ho visto a catalogo.",
            "Potete inviarmi i dettagli dell'offerta speciale di cui mi avete parlato al telefono?"
        ]
    }
}

# Frasi aggiuntive per generare variabilità e testare le regole di priorità
urgency_phrases = ["È urgente.", "Situazione bloccante.", "Produzione ferma.", "Risolvete subito.", "ASAP!"]
low_priority_phrases = ["Solo per informazione.", "Quando avete tempo.", "Nessuna fretta.", "Solo per avere un'idea."]


def generate_ticket(ticket_id):
    """Genera un singolo ticket sintetico."""
    category = random.choice(list(data_templates.keys()))
    title = random.choice(data_templates[category]["titles"])
    body = random.choice(data_templates[category]["bodies"])

    # Introduciamo variabilità: il 25% dei ticket avrà frasi di urgenza, il 25% di bassa priorità
    prob = random.random()
    if prob < 0.25:
        body += " " + random.choice(urgency_phrases)
    elif prob < 0.50:
        body += " " + random.choice(low_priority_phrases)

    # 2. Logica di assegnazione Priorità basata su parole chiave (rule-based)
    testo_completo = (title + " " + body).lower()

    # Liste di parole chiave per la classificazione della priorità
    alta_kw = ["bloccante", "urgente", "ferma", "subito", "crash", "errore 500", "asap"]
    bassa_kw = ["informazione", "tempo", "fretta", "idea", "preventivo"]

    # Assegnazione priorità
    if any(kw in testo_completo for kw in alta_kw):
        priority = "Alta"
    elif any(kw in testo_completo for kw in bassa_kw):
        priority = "Bassa"
    else:
        priority = "Media"

    return {
        "id": f"TCK-{ticket_id:04d}",
        "title": title,
        "body": body,
        "category": category,
        "priority": priority
    }


# 3. Generiamo 300 ticket (Soddisfa il requisito 200-500)
num_tickets = 300
tickets = [generate_ticket(i) for i in range(1, num_tickets + 1)]

# 4. Creiamo il DataFrame
df = pd.DataFrame(tickets)

# --- INIZIO MODIFICA PER I PERCORSI ---
# Trova il percorso assoluto della cartella in cui si trova questo script (src)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Risali di un livello per arrivare alla cartella principale del progetto
project_root = os.path.dirname(script_dir)

# Costruisci il percorso corretto per la cartella data/
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True) # Crea data/ nella root del progetto

csv_path = os.path.join(data_dir, "synthetic_tickets.csv")
# --- FINE MODIFICA ---

# Salva il file
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"✅ Dataset generato con successo!")
print(f"📁 Salvato esattamente in: {csv_path}")
print(f"📊 Totale righe: {len(df)}")
print("\nPrime 3 righe generate:")
print(df.head(3).to_string(index=False))
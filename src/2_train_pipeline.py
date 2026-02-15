import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# --- 1. GESTIONE PERCORSI SICURA (stile PyCharm) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_path = os.path.join(project_root, "data", "synthetic_tickets.csv")
models_dir = os.path.join(project_root, "models")
reports_dir = os.path.join(project_root, "reports") # Qui salveremo i grafici per il PDF

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# --- 2. CARICAMENTO DATI E PREPROCESSING ---
print("Caricamento dataset...")
df = pd.read_csv(data_path)

# Combiniamo Titolo e Corpo per dare al modello tutto il contesto
df['text'] = df['title'] + " " + df['body']

# Separiamo le feature (X) dai target (y)
X = df['text']
y_cat = df['category']
y_prio = df['priority']

# Train/Test Split (80% addestramento, 20% valutazione) come richiesto
X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
    X, y_cat, y_prio, test_size=0.2, random_state=42
)

# --- 3. VETTORIZZAZIONE (Testo -> Numeri) ---
print("Vettorizzazione del testo con TF-IDF...")
# TfidfVectorizer fa in automatico il preprocessing base:
# converte in minuscolo e rimuove la punteggiatura tramite tokenizzazione.
vectorizer = TfidfVectorizer(max_features=1000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 4. TRAINING E VALUTAZIONE: CATEGORIA ---
print("\n--- Addestramento Modello CATEGORIA ---")
clf_cat = LogisticRegression(random_state=42)
clf_cat.fit(X_train_vec, y_cat_train)

y_cat_pred = clf_cat.predict(X_test_vec)
cat_acc = accuracy_score(y_cat_test, y_cat_pred)
cat_f1 = f1_score(y_cat_test, y_cat_pred, average='macro')
print(f"Categoria - Accuracy: {cat_acc:.2f} | F1 Macro: {cat_f1:.2f}")

# Plot Confusion Matrix Categoria
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_cat_test, y_cat_pred, ax=ax, cmap='Blues')
plt.title("Confusion Matrix - Categoria")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "cm_categoria.png"))
plt.close()

# --- 5. TRAINING E VALUTAZIONE: PRIORITÀ ---
print("\n--- Addestramento Modello PRIORITÀ ---")
clf_prio = LogisticRegression(random_state=42)
clf_prio.fit(X_train_vec, y_prio_train)

y_prio_pred = clf_prio.predict(X_test_vec)
prio_acc = accuracy_score(y_prio_test, y_prio_pred)
prio_f1 = f1_score(y_prio_test, y_prio_pred, average='macro')
print(f"Priorità  - Accuracy: {prio_acc:.2f} | F1 Macro: {prio_f1:.2f}")

# Plot Confusion Matrix Priorità
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_prio_test, y_prio_pred, ax=ax, cmap='Oranges')
plt.title("Confusion Matrix - Priorità")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "cm_priorita.png"))
plt.close()

# --- 6. SALVATAGGIO MODELLI ---
print("\nSalvataggio modelli nella cartella 'models/'...")
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
joblib.dump(clf_cat, os.path.join(models_dir, "model_category.pkl"))
joblib.dump(clf_prio, os.path.join(models_dir, "model_priority.pkl"))

print("✅ Pipeline completata con successo! Grafici e modelli salvati.")
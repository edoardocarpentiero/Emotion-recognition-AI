import os

def get_error_filenames(error_root):
    """Raccoglie i nomi di tutti i file presenti nelle sotto-cartelle di errors/"""
    filenames = set()
    print("[DEBUG] Inizio scansione delle cartelle di errore...")
    for subfolder in os.listdir(error_root):
        subfolder_path = os.path.join(error_root, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"[DEBUG] Controllo cartella: {subfolder_path}")
            for fname in os.listdir(subfolder_path):
                if fname.lower().endswith((".jpg", ".png")):
                    filenames.add(fname)
                    print(f"[DEBUG] Trovata immagine ambigua: {fname}")
    print(f"[DEBUG] Totale immagini ambigue trovate: {len(filenames)}")
    return filenames

def delete_matching_files(train_root, error_filenames):
    """Elimina i file in assets/train che corrispondono ai nomi raccolti da errors/"""
    print(f"ErroFilenasme: {error_filenames}")
    deleted_count = 0
    print("\n[DEBUG] Inizio scansione cartelle di training per eliminare immagini...")
    for subfolder in os.listdir(train_root):
        subfolder_path = os.path.join(train_root, subfolder)
        if os.path.isdir(subfolder_path):

            for fname in os.listdir(subfolder_path):
                print(f"[DEBUG] {fname} -> {subfolder_path}")
                if fname in error_filenames:
                    file_path = os.path.join(subfolder_path, fname)
                    print(f"[DEBUG] Controllo cartella: {subfolder_path}")
                    try:
                        os.remove(file_path)
                        print(f"[✔] Eliminata: {file_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"[✘] Errore nell'eliminazione di {file_path}: {e}")
                #else:
                #    print(f"[DEBUG] Nessuna corrispondenza per: {fname}")
    print(f"\n[INFO] Totale immagini eliminate: {deleted_count}")

# === CONFIGURA QUI I PERCORSI ===
errors_dir = os.path.abspath("errors")             # es. "./errors"
assets_train_dir = os.path.abspath("assets/dataset_bilanciato") # es. "./assets/train"

# === ESECUZIONE ===
print("[INFO] Avvio script di pulizia immagini ambigue...")
error_filenames = get_error_filenames(errors_dir)
delete_matching_files(assets_train_dir, error_filenames)
print("[INFO] Operazione completata.")

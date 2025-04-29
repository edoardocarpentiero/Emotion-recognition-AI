import os
import shutil

def remove_directory_if_exists(directory_path):
    # Verifica se la directory esiste
    if os.path.exists(directory_path):
        if os.path.isdir(directory_path):  # Verifica che sia una directory
            try:
                if not os.listdir(directory_path):  # Se la directory è vuota
                    os.rmdir(directory_path)  # Rimuove la directory vuota
                    print(f"Directory vuota {directory_path} rimossa con successo!")
                else:
                    shutil.rmtree(directory_path)  # Rimuove la directory non vuota e tutto il suo contenuto
                    print(f"Directory non vuota {directory_path} e il suo contenuto sono stati rimossi!")
            except Exception as e:
                print(f"Errore durante la rimozione: {e}")
        else:
            print(f"{directory_path} non è una directory!")
    else:
        print(f"Directory {directory_path} non esiste!")


def empty_directory(directory_path):
    # Verifica che la directory esista
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Elenco dei file e sottodirectory nella directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            try:
                # Se è un file, lo rimuovi
                if os.path.isfile(file_path):
                    os.remove(file_path)
                # Se è una sottodirectory, la rimuovi ricorsivamente
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Rimuove solo directory vuote
                print(f"Rimosso: {file_path}")
            except Exception as e:
                print(f"Errore durante la rimozione di {file_path}: {e}")
    else:
        print(f"La directory {directory_path} non esiste o non è una directory!")

def cleanFolder(split_dir_balanced,train_dir_balanced):
    remove_directory_if_exists(split_dir_balanced)
    remove_directory_if_exists(train_dir_balanced)
    remove_directory_if_exists('assets/train_fused')
    empty_directory('results/errors')

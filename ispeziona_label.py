from pathlib import Path

BASE = Path("data/rs19_val")
label_path = BASE / "labels" / "rs00000.txt"

righe = label_path.read_text().strip().split("\n")
print(f"Totale righe (oggetti) nel file: {len(righe)}\n")

# conta per classe e mostra quanti punti ha ogni rotaia
from collections import defaultdict
per_classe = defaultdict(list)

for riga in righe:
    if not riga.strip():
        continue
    parti = riga.split()
    classe = int(parti[0])
    n_punti = (len(parti) - 1) // 2   # ogni punto = 2 valori (x,y)
    per_classe[classe].append(n_punti)

nomi = {0: "rotaie (rail-raised)", 1: "pali", 2: "vegetazione"}
for classe in sorted(per_classe):
    lista = per_classe[classe]
    print(f"Classe {classe} ({nomi.get(classe,'?')}): {len(lista)} oggetti, punti per oggetto: {lista}")
#!/usr/bin/env python3
"""Valutazione di uno o piu' modelli YOLOv8 sul test set del dataset di grasping.

Di default valuta TUTTI i modelli in MODEL_PATHS, uno alla volta, sulle
STESSE 98 immagini di test (data/training_dataset.yolov8/test/ -- l'unico
dataset "completo" presente nel repo, usato per tutto il progetto finora).

Per valutarne uno solo, YOLO_MODEL_PATH sovrascrive MODEL_PATHS (comodo
mentre si aggiunge/tara un modello nuovo, senza rilanciare tutti gli
altri):
  YOLO_MODEL_PATH=/path/al/modello.pt python3 evaluate.py

--- Perche' il ground truth viene filtrato e rimappato per modello ---

I modelli in MODEL_PATHS NON hanno tutti le stesse classi del dataset
completo (6: aruco marker, biscuits pack, bookshelf, coke can,
dinner table, pringles can). Es. "small_omogeneous_dataset_model" ne
conosce UNA sola (coke can); il modello di segmentazione ne conosce 3
(coke can, pringles can, biscuits pack -- niente bookshelf/dinner
table/aruco marker). Passare il ground truth a 6 classi cosi' com'e' a
model.val() non e' solo scorretto (penalizzerebbe un modello mono-classe
per non aver rilevato una libreria che non ha mai imparato a riconoscere)
-- CRASHA: Ultralytics costruisce la confusion matrix dimensionata sul
numero di classi del MODELLO, e un oggetto ground truth di una classe
sconosciuta al modello (indice fuori range) manda l'array fuori dai
limiti (IndexError, verificato).

Per ogni modello si ricostruisce quindi un mini dataset di valutazione
(build_filtered_test_dataset, in una cartella temporanea): stesse
immagini (simlink, nessuna duplicazione), ma le label filtrate a runtime
per contenere SOLO gli oggetti delle classi che il modello conosce
(matchate per nome, dopo normalize_yolo_class_name -- vedi li' il perche'
serve: run di training diverse hanno etichettato le stesse classi in modo
leggermente diverso, es. 'coke_can' con underscore invece di uno spazio),
con l'indice di classe rimappato dall'indice del dataset completo
all'indice proprio del modello. Per un modello con tutte e 6 le classi
nello stesso ordine del dataset (es. "small_eterogeneous") il filtro e'
un'identita' -- nessuna differenza rispetto a valutare sul dataset intero.

--- Perche' un modello di SEGMENTAZIONE viene valutato in modalita' 'detect' ---

"segmentation_model_best" (YOLO26s-seg, non YOLOv8 come gli altri 3 --
architettura diversa, non solo taglia) predice anche maschere per
istanza, non solo box. Il nostro ground truth filtrato sopra e' pero' in
formato box puro (classe x y w h): passato cosi' com'e' al validator di
segmentazione, che si aspetta poligoni, genera maschere degeneri e
crasha (verificato: "The shape of the mask [83] at index 0 does not
match the shape of the indexed tensor [0, 1] at index 0"). Per questo
modello si costruisce quindi un'istanza YOLO separata con
task='detect' forzato alla creazione (l'override non ha effetto passato
a .val(): il task e' fissato alla costruzione del Model) -- stessi pesi,
ma Ultralytics userà il Validator box-only, coerente col nostro ground
truth. La metrica di segmentazione (mask mAP) non viene quindi
calcolata per questo modello, solo quella box -- comunque quella che
interessa per il grasping (target/ostacoli localizzati da un box, non
da una maschera pixel-precisa).

Ogni modello stampa un'intestazione con path e classi PRIMA dei risultati
(altrimenti, con piu' modelli in sequenza, l'output non direbbe a quale
modello si riferisce) e scrive in una propria sottocartella dentro
models_evaluation/models_comparison/<nome_checkpoint>/ -- cosi' i
risultati di un modello non sovrascrivono quelli del precedente. Ogni
sottocartella contiene SOLO i dati di quel modello (evaluation/,
prediction/): nessun grafico di confronto incrociato generato da questo
script -- per quello vedi models_comparison_sweep.py (solo run dello
sweep W&B pero', non questi 4 checkpoint).

NOTA: l'output vive in models_evaluation/, non in models/ -- quella
cartella resta riservata ai pesi veri (wandb/, pretrained/, i checkpoint
*.pt), tutto cio' che questo script genera (grafici, confusion matrix,
immagini annotate) e' derivato/rigenerabile, non un modello.
"""
import os
import tempfile

import yaml
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
MODEL_PATHS = [
    "/home/user/ros_workspace/src/vision_pipeline/models/wandb/runs/sqkfh2ka/training/xl1874f6/weights/best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/small_omogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/small_eterogeneous_dataset_model_best.pt",
    "/home/user/ros_workspace/src/vision_pipeline/models/segmentation_model_best.pt",
]
if 'YOLO_MODEL_PATH' in os.environ:
    MODEL_PATHS = [os.environ['YOLO_MODEL_PATH']]  # un solo modello, non tutti e 4

DATA_YAML = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/data.yaml"
TEST_IMAGES_DIR = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/images"
TEST_LABELS_DIR = "/home/user/ros_workspace/src/vision_pipeline/data/training_dataset.yolov8/test/labels"
# NON dentro models/ (riservata ai pesi veri) -- vedi nota in cima al file.
EVALUATION_DIR = "/home/user/ros_workspace/src/vision_pipeline/models_evaluation"

# Le classi che contano davvero per il task di grasping (posa target end effector).
# bookshelf e dinner table sono contesto di scena, non oggetti da afferrare/riferimento di posa.
GRASP_RELEVANT_CLASSES = ["coke can", "pringles can", "biscuits pack", "aruco marker"]


def normalize_yolo_class_name(raw_class_name):
    """
    Stessa normalizzazione usata in rt_object_detection_node_all.py (vedi
    li' per il perche' serve): underscore -> spazio, spazi multipli -> uno
    solo. Trovato confrontando questi stessi modelli: run di training
    diverse etichettano leggermente diverso anche per le STESSE classi
    (es. 'coke_can' invece di 'coke can', 'pringles  can' con due spazi).
    """
    return ' '.join(raw_class_name.replace('_', ' ').split())


def load_original_class_names():
    """Nomi classe del dataset completo, in ordine di indice (0..5)."""
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    return data['names']


def build_filtered_test_dataset(model_class_names, original_class_names, tmp_dir):
    """
    Ricostruisce, dentro tmp_dir, un mini dataset di valutazione per un
    modello con classi (model_class_names, indicizzate come nel modello)
    diverse -- in numero e/o ordine -- da quelle del dataset completo
    (original_class_names, indicizzate come nel dataset). Vedi il
    docstring del modulo per il perche'.

    Ritorna il path al data.yaml risultante.
    """
    original_name_to_index = {
        normalize_yolo_class_name(name): idx for idx, name in enumerate(original_class_names)
    }
    # indice ORIGINALE (dataset) -> indice del MODELLO, solo per le classi
    # che il modello conosce davvero (matchate per nome normalizzato).
    original_to_model_index = {}
    for model_idx, raw_name in enumerate(model_class_names):
        name = normalize_yolo_class_name(raw_name)
        original_idx = original_name_to_index.get(name)
        if original_idx is not None:
            original_to_model_index[original_idx] = model_idx
        # Se una classe del modello non esiste proprio nel dataset di
        # riferimento (non dovrebbe succedere per questi 4 modelli, sono
        # tutti addestrati su varianti dello stesso set di oggetti), resta
        # semplicemente senza nessun esempio positivo nel ground truth
        # filtrato -- val() la tratta come classe senza supporto, non crasha.

    images_dir = os.path.join(tmp_dir, 'test', 'images')
    labels_dir = os.path.join(tmp_dir, 'test', 'labels')
    os.makedirs(os.path.dirname(images_dir), exist_ok=True)
    os.symlink(TEST_IMAGES_DIR, images_dir)  # niente duplicazione delle immagini vere
    os.makedirs(labels_dir, exist_ok=True)

    label_filenames = [f for f in os.listdir(TEST_LABELS_DIR) if f.endswith('.txt')]
    for filename in label_filenames:
        with open(os.path.join(TEST_LABELS_DIR, filename)) as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            original_class_idx = int(parts[0])
            model_class_idx = original_to_model_index.get(original_class_idx)
            if model_class_idx is not None:
                filtered_lines.append(' '.join([str(model_class_idx)] + parts[1:]) + '\n')
            # riga scartata: oggetto di una classe che il modello non conosce --
            # esattamente il filtro che serve (vedi docstring del modulo).

        # Scritta comunque (anche vuota = immagine "background" per questo
        # modello): Ultralytics si aspetta un file per ogni immagine.
        with open(os.path.join(labels_dir, filename), 'w') as f:
            f.writelines(filtered_lines)

    data_yaml_path = os.path.join(tmp_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.safe_dump({
            # train/val non vengono mai usati (solo split="test" sotto),
            # ma servono path validi perche' Ultralytics valida il file.
            'train': images_dir,
            'val': images_dir,
            'test': images_dir,
            'nc': len(model_class_names),
            'names': [normalize_yolo_class_name(n) for n in model_class_names],
        }, f)

    return data_yaml_path


def model_stem_for(model_path):
    """
    Nome usato per la sottocartella di output e nei log. Di norma il nome
    file senza estensione, ma "best.pt" e' il nome generico che Ultralytics
    da' SEMPRE al checkpoint migliore di una run (qualunque run) -- usarlo
    cosi' com'e' sarebbe ambiguo (di quale run e' il "best"?). Tra i
    modelli di MODEL_PATHS e' il checkpoint dello sweep W&B: rinominato
    esplicitamente per dirlo.
    """
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return 'sweep_best' if stem == 'best' else stem


def evaluate_one_model(model_path, original_class_names):
    model_stem = model_stem_for(model_path)
    model_output_dir = os.path.join(EVALUATION_DIR, "models_comparison", model_stem)

    model = YOLO(model_path)
    model_class_names = [model.names[i] for i in sorted(model.names)]
    model_class_names_normalized = [normalize_yolo_class_name(n) for n in model_class_names]

    print("\n" + "#" * 60)
    print(f"# MODELLO: {model_stem}")
    print(f"# Path: {model_path}")
    print(f"# Task: {model.task}")
    print(f"# Classi: {', '.join(model_class_names_normalized)}")
    print(f"# Output: {model_output_dir}")
    print("#" * 60)

    # Un modello di SEGMENTAZIONE (es. segmentation_model_best, YOLO26s-seg)
    # va valutato in modalita' 'detect' (solo il ramo box, maschera
    # ignorata): il nostro ground truth filtrato (build_filtered_test_dataset)
    # e' in formato box puro (classe x y w h), non poligoni -- passato al
    # validator di segmentazione cosi' com'e' genera maschere degeneri e
    # crasha (verificato: "The shape of the mask [83] at index 0 does not
    # match the shape of the indexed tensor [0, 1] at index 0"). Serve
    # un'istanza YOLO separata con task forzato a costruzione (l'override
    # non funziona passato a .val(), il task e' fissato alla creazione
    # dell'oggetto Model) -- model_for_val resta comunque lo stesso
    # checkpoint/pesi, cambia solo quale Validator Ultralytics sceglie.
    model_for_val = YOLO(model_path, task='detect') if model.task == 'segment' else model

    # ─────────────────────────────────────────────
    # VALUTAZIONE SUL TEST SET, FILTRATO/RIMAPPATO SULLE CLASSI DEL MODELLO
    # (vedi il docstring del modulo per il perche')
    # ─────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix=f'yolo_eval_{model_stem}_') as tmp_dir:
        filtered_data_yaml = build_filtered_test_dataset(
            model_class_names, original_class_names, tmp_dir
        )

        try:
            metrics = model_for_val.val(
                data=filtered_data_yaml,
                split="test",
                save_json=True,     # salva anche i risultati in formato COCO json
                plots=True,          # genera confusion_matrix.png, PR/F1/P/R curves, ecc.
                project=model_output_dir,
                name="evaluation",
                exist_ok=True,
            )
        except Exception as error:
            # Difesa in profondita': il filtro sopra elimina la causa nota
            # del crash (indice fuori range), ma se dovesse comunque
            # fallire per un altro motivo non blocchiamo gli altri modelli
            # -- procediamo comunque con l'inferenza visiva sotto.
            print(f"\n{'!' * 60}")
            print(f"ATTENZIONE: valutazione (mAP/confusion matrix) fallita per {model_stem}: {error}")
            print("Procedo comunque con l'inferenza visiva (prediction/).")
            print("!" * 60)
            metrics = None

    if metrics is not None:
        # metrics.box.maps è un array numpy con la mAP50-95 per ciascuna classe (stesso ordine di model_class_names)
        maps_per_class = metrics.box.maps

        # metrics.box.p, metrics.box.r, metrics.box.ap50 sono array allineati con ap_class_index.
        # Nota: se una classe non compare nel test set filtrato, questi array potrebbero avere
        # lunghezza minore del numero di classi del modello — in quel caso Ultralytics riporta
        # le classi effettivamente valutate nell'ordine di metrics.box.ap_class_index.
        ap_class_index = metrics.box.ap_class_index  # indici (nello spazio del MODELLO) delle classi presenti
        precision_per_class = metrics.box.p
        recall_per_class = metrics.box.r
        ap50_per_class = metrics.box.ap50

        # ─────────────────────────────────────────────
        # STAMPA METRICHE AGGREGATE (riferimento generale, non l'obiettivo primario)
        # ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"METRICHE AGGREGATE (sulle {len(model_class_names)} classi di questo modello)")
        print("=" * 60)
        print(f"mAP50:     {metrics.box.map50:.3f}")
        print(f"mAP50-95:  {metrics.box.map:.3f}")
        print(f"Precision: {metrics.box.p.mean():.3f}")
        print(f"Recall:    {metrics.box.r.mean():.3f}")

        # ─────────────────────────────────────────────
        # STAMPA METRICHE PER CLASSE (diagnostica)
        # ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("METRICHE PER CLASSE")
        print("=" * 60)
        print(f"{'Classe':<20}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
        print("-" * 68)

        per_class_map5095 = {}
        for i, class_idx in enumerate(ap_class_index):
            name = model_class_names_normalized[int(class_idx)]
            p = precision_per_class[i]
            r = recall_per_class[i]
            ap50 = ap50_per_class[i]
            ap5095 = maps_per_class[int(class_idx)]
            per_class_map5095[name] = ap5095
            print(f"{name:<20}{p:>12.3f}{r:>12.3f}{ap50:>12.3f}{ap5095:>12.3f}")

        # ─────────────────────────────────────────────
        # FUNZIONE OBIETTIVO: mAP50-95 media sulle classi rilevanti per il grasping
        # CHE QUESTO MODELLO CONOSCE (le altre non hanno senso per lui)
        # ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("FUNZIONE OBIETTIVO — mAP50-95 (classi rilevanti per grasping)")
        print("=" * 60)

        grasp_maps = []
        for name in GRASP_RELEVANT_CLASSES:
            if name not in model_class_names_normalized:
                print(f"  {name:<20}: il modello non conosce questa classe")
            elif name in per_class_map5095:
                val = per_class_map5095[name]
                grasp_maps.append(val)
                print(f"  {name:<20}: {val:.3f}")
            else:
                print(f"  {name:<20}: non presente nel test set (attenzione, controlla il dataset)")

        if grasp_maps:
            grasp_mean = sum(grasp_maps) / len(grasp_maps)
            print(f"\n>>> mAP50-95 media (grasping-relevant, sulle classi note a questo modello): {grasp_mean:.3f}  <<<")
            print("(confrontabile con altri modelli solo se conoscono le stesse classi grasping-relevant)")
        else:
            print("\nATTENZIONE: nessuna delle classi grasping-relevant è nota a questo modello o nel test set.")

        # ─────────────────────────────────────────────
        # DOVE TROVARE I PLOT GENERATI DA ULTRALYTICS
        # ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("PLOT E FILE GENERATI")
        print("=" * 60)
        print(f"Cartella: {metrics.save_dir}")
        print("  - confusion_matrix.png")
        print("  - confusion_matrix_normalized.png")
        print("  - BoxP_curve.png   (Precision-Confidence)")
        print("  - BoxR_curve.png   (Recall-Confidence)")
        print("  - BoxF1_curve.png  (F1-Confidence)")
        print("  - BoxPR_curve.png  (Precision-Recall)")
        print("  - val_batch*_labels.jpg / val_batch*_pred.jpg  (campione a griglia, poche immagini)")
        print("  - predictions.json (risultati in formato COCO, se save_json=True)")

    # ─────────────────────────────────────────────
    # INFERENCE SU TUTTE LE IMMAGINI DI TEST CON BOUNDING BOX DISEGNATI
    # (una immagine di output per ciascuna immagine di input, non solo il campione a griglia sopra;
    # non dipende dal ground truth, gira sempre anche se la valutazione sopra e' fallita)
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENCE CON BOUNDING BOX SU TUTTE LE IMMAGINI DI TEST")
    print("=" * 60)

    pred_results = model.predict(
        source=TEST_IMAGES_DIR,
        conf=0.25,
        save=True,
        show_labels=True,
        show_conf=True,
        project=model_output_dir,
        name="prediction",
        exist_ok=True,
    )

    # Stampa i risultati per ogni immagine, come nello script di inferenza originale
    for r in pred_results:
        img_name = os.path.basename(r.path)
        print(f"\nImmagine: {img_name}")
        if len(r.boxes) == 0:
            print("  → nessun oggetto rilevato")
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = model.names[cls]
            print(f"  → {name}: {conf*100:.1f}%")

    print(f"\nImmagini con bounding box salvate in: {model_output_dir}/prediction/")


def main():
    original_class_names = load_original_class_names()

    print(f"Modelli da valutare ({len(MODEL_PATHS)}):")
    for path in MODEL_PATHS:
        print(f"  - {path}")

    for model_path in MODEL_PATHS:
        evaluate_one_model(model_path, original_class_names)

    print("\n" + "#" * 60)
    print(f"# Fatto -- {len(MODEL_PATHS)} modelli valutati.")
    print(f"# Risultati in: {EVALUATION_DIR}/models_comparison/<nome_checkpoint>/")
    print("#" * 60)


if __name__ == '__main__':
    main()

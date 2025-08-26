# main.py
import os
import re
import time
from stl_loader import load_stl
from transform import align_models
from visualizer import visualize_pairs

DATA_DIR = "data"

def extract_id(name: str):
    """
    Возвращает целочисленный префикс из начала имени файла.
    Примеры: '1_upper.stl'->1, '10-низ.stl'->10, '028_top.stl'->28.
    """
    m = re.match(r'^0*(\d+)', name)
    return int(m.group(1)) if m else None

def detect_role(filename: str):
    """
    Определяем роль по имени (рус/англ/сокр.): 'upper/верх/top/_u' и 'lower/низ/bottom/_l'.
    Возвращает 'upper' | 'lower' | None.
    """
    n = filename.lower()
    if any(k in n for k in ('upper', 'верх', 'verh', 'top')):
        return 'upper'
    if any(k in n for k in ('lower', 'низ', 'niz', 'bottom')):
        return 'lower'
    m = re.search(r'_(u|l)\b', n)
    if m:
        return 'upper' if m.group(1) == 'u' else 'lower'
    return None

def collect_pairs_from_folder(folder_path: str):
    """
    Собираем ПОЛНЫЕ пары по ПОЛНОМУ числовому префиксу:
      1_upper.stl + 1_lower.stl, 10_*.stl + 10_*.stl, 28_*.stl + 28_*.stl и т.д.
    Возвращает список (upper_mesh, lower_mesh, pair_id:int), отсортированный по id.
    """
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.stl')]
    buckets = {}  # id -> {"upper":[...], "lower":[...]}

    for fn in files:
        pid = extract_id(fn)
        if pid is None:
            print(f"[skip] нет числового префикса: {fn}")
            continue
        role = detect_role(fn)
        if role not in ('upper', 'lower'):
            print(f"[skip] не удалось определить роль (upper/lower): {fn}")
            continue
        buckets.setdefault(pid, {"upper": [], "lower": []})[role].append(fn)

    pairs = []
    for pid in sorted(buckets.keys()):
        group = buckets[pid]
        if not group["upper"] or not group["lower"]:
            miss = "upper" if not group["upper"] else "lower"
            print(f"[warn] id={pid}: нет {miss} — пара пропущена")
            continue

        # если дублей несколько — берём первый; при необходимости можно выбрать по правилу
        up_name = group["upper"][0]
        low_name = group["lower"][0]

        try:
            upper_mesh = load_stl(up_name)
            lower_mesh = load_stl(low_name)
        except Exception as e:
            print(f"[error] загрузка id={pid}: {e}")
            continue

        if upper_mesh is None or lower_mesh is None:
            print(f"[warn] id={pid}: пустая сетка — пропуск")
            continue

        pairs.append((upper_mesh, lower_mesh, pid))

    print(f"[info] найдено пар: {len(pairs)} из файлов: {len(files)}")
    return pairs

def main():
    folder_path = DATA_DIR
    pairs = collect_pairs_from_folder(folder_path)
    if not pairs:
        print("Нет пар для отображения")
        return

    plotter = visualize_pairs(pairs)

    # Вариант A: обычный pv.Plotter (блокирующий)
    if not hasattr(plotter, "app") or plotter.app is None:
        plotter.show()
        return

    # Вариант B: Qt-плоттер — мягкий цикл, чтобы IDE не падала на exec_()
    app = plotter.app
    try:
        while True:
            if hasattr(app, "closingDown") and app.closingDown():
                break
            app.processEvents()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
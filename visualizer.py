# visualizer.py
import pyvista as pv
import numpy as np

# --- попытка импортировать Qt-плоттер
try:
    from pyvistaqt import BackgroundPlotter as _BackgroundPlotter
except Exception:
    _BackgroundPlotter = None

# --- необязательная прослойка для Qt (чтобы не тянуть напрямую PyQt5/PySide)
try:
    from qtpy import QtCore, QtWidgets
except Exception:
    QtCore = QtWidgets = None


def get_center_of_mass(mesh):
    return np.array(mesh.center)


def get_bounding_box_size(mesh):
    b = mesh.bounds
    return np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])


def normalize_scale(upper_mesh, lower_mesh):
    upper_size = get_bounding_box_size(upper_mesh)
    lower_size = get_bounding_box_size(lower_mesh)
    lower_size = np.where(lower_size == 0, 1e-9, lower_size)  # защита от нуля
    scale_factor = upper_size / lower_size
    return lower_mesh.scale(scale_factor)


def _prep_pair(upper_mesh, lower_mesh):
    """Готовим пару к показу: масштаб, совмещение центров, твои сдвиги."""
    lower_mesh = normalize_scale(upper_mesh, lower_mesh)

    upper_center = get_center_of_mass(upper_mesh)
    lower_center = get_center_of_mass(lower_mesh)

    translation = upper_center - lower_center
    translation[1] -= 8  # твой сдвиг по Y
    lower_mesh = lower_mesh.translate(translation)

    # твои смещения верхней
    upper_mesh = upper_mesh.translate([-1, +1, +3])
    return upper_mesh, lower_mesh


def visualize_pairs(pairs):
    """
    Показ пар без автоплея. НИКАКИХ VTK-виджетов.
    Управление:
      - Везде: клавиши A (prev) / D (next)
      - Если есть pyvistaqt: тулбар Qt с кнопками Prev/Next и слайдером
    Возвращает plotter; окно не запускается здесь.
    """
    plotter = _BackgroundPlotter(show=True) if _BackgroundPlotter else pv.Plotter()
    plotter.set_background("white")

    # ---- создаём акторы заранее и скрываем (без мерцаний)
    actors = []  # [(actor_upper, actor_lower, pair_id), ...]
    for upper_mesh, lower_mesh, pid in pairs:
        up, low = _prep_pair(upper_mesh.copy(), lower_mesh.copy())
        a_up = plotter.add_mesh(up, color='blue', opacity=1.0, label=f"Upper #{pid}")
        a_low = plotter.add_mesh(low, color='red', opacity=1.0, label=f"Lower #{pid}")
        a_up.SetVisibility(False)
        a_low.SetVisibility(False)
        actors.append((a_up, a_low, pid))

    plotter.add_legend()

    state = {"idx": 0}

    def _clamp_index(i: int) -> int:
        n = len(actors)
        if n == 0:
            return 0
        return max(0, min(n - 1, i))

    def show_index(idx: int):
        try:
            plotter.renderer.enable_depth_peeling = True
        except Exception:
            pass

        for a_up, a_low, _ in actors:
            a_up.SetVisibility(False)
            a_low.SetVisibility(False)

        a_up, a_low, pid = actors[idx]
        a_up.SetVisibility(True)
        a_low.SetVisibility(True)

        hud_text = f"Пара ID: {pid}   [{idx + 1}/{len(actors)}]   A — Prev, D — Next"
        plotter.add_text(hud_text, name="hud", position='upper_left', font_size=10)
        plotter.render()

    def goto_index(new_idx: int):
        state["idx"] = _clamp_index(new_idx)
        show_index(state["idx"])

    def next_pair(*_):
        goto_index(state["idx"] + 1)

    def prev_pair(*_):
        goto_index(state["idx"] - 1)

    # стартовый кадр
    if actors:
        show_index(state["idx"])
    else:
        plotter.add_text("Нет пар для отображения", name="hud", position='upper_left', font_size=10)

    # --- Горячие клавиши: только пошагово
    if hasattr(plotter, "add_key_event"):
        plotter.add_key_event('a', lambda: prev_pair())
        plotter.add_key_event('d', lambda: next_pair())

    # --- Qt-тулбар вместо VTK-виджетов (без ложных триггеров)
    def build_qt_toolbar_once():
        if not (_BackgroundPlotter and QtWidgets and hasattr(plotter, "app_window")):
            return
        win = plotter.app_window

        tb = getattr(win, "_pairs_toolbar", None)
        if tb is not None:
            return  # уже создан

        tb = win.addToolBar("Pairs")
        win._pairs_toolbar = tb

        # Кнопка Prev
        act_prev = tb.addAction("⟵ Prev")
        act_prev.triggered.connect(lambda: prev_pair())

        # Кнопка Next
        act_next = tb.addAction("Next ⟶")
        act_next.triggered.connect(lambda: next_pair())

        tb.addSeparator()

        # Слайдер индекса (реагирует только по отпусканию ползунка)
        if len(actors) > 0:
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, tb)
            slider.setMinimum(0)
            slider.setMaximum(len(actors) - 1)
            slider.setValue(state["idx"])
            slider.setTickInterval(1)
            slider.setSingleStep(1)
            slider.setFixedWidth(220)

            label = QtWidgets.QLabel(f"{state['idx']+1}/{len(actors)}", tb)

            def _on_move(val):
                label.setText(f"{val+1}/{len(actors)}")

            def _on_released():
                goto_index(slider.value())

            slider.valueChanged.connect(_on_move)
            slider.sliderReleased.connect(_on_released)

            tb.addWidget(slider)
            tb.addWidget(label)

    # Построить тулбар, если это Qt-ветка
    build_qt_toolbar_once()

    return plotter
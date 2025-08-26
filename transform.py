import pyvista as pv

def align_models(upper_mesh, lower_mesh):
    """Функция для выравнивания моделей верхней и нижней челюсти."""
    # Пример выравнивания по оси Z
    lower_mesh.points[:, 2] -= lower_mesh.points[:, 2].min()  # Сдвигаем по оси Z

    # Подгонка нижней челюсти под верхнюю по оси Z
    translation = [0, 0, upper_mesh.points[:, 2].max() - lower_mesh.points[:, 2].min()]
    lower_mesh.translate(translation)

    return lower_mesh
import pyvista as pv
import os

def load_stl(file_name):
    """Загружает STL файл из папки data."""
    file_path = os.path.join('data', file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_name} не найден в папке data.")
    return pv.read(file_path)
from nbconvert import ScriptExporter

def convert_to_python(input_notebook, output_script):
    """
    Convierte un archivo de Jupyter Notebook a un archivo de script de Python.

    Parámetros:
    input_notebook (str): La ruta al archivo de Jupyter Notebook de entrada (.ipynb).
    output_script (str): La ruta al archivo de script de Python de salida (.py).
    """
    # Inicializar el exportador de scripts
    exporter = ScriptExporter()

    # Leer el contenido del notebook
    with open(input_notebook, 'r', encoding='utf-8') as f:
        notebook_content = f.read()

    # Convertir el notebook a un script de Python
    python_script, _ = exporter.from_filename(input_notebook)

    # Escribir el script de Python en el archivo de salida
    with open(output_script, 'w', encoding='utf-8') as f:
        f.write(python_script)

# Ejemplo de uso:
if __name__ == "__main__":
    input_notebook = 'Prediction lab.ipynb'  # Ruta al archivo .ipynb de entrada
    output_script = 'Prediction lab.py'  # Ruta al archivo .py de salida
    convert_to_python(input_notebook, output_script)

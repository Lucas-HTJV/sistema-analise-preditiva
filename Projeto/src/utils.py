from tkinter import Tk, filedialog
from typing import Optional, List, Tuple


def selecionar_arquivo(
    titulo: str = "Selecione um arquivo",
    tipos: Optional[List[Tuple[str, str]]] = None
) -> Optional[str]:
    """
    Abre uma janela para seleção de arquivo.

    Parameters
    ----------
    titulo : str
        Título da janela.
    tipos : list of tuples
        Tipos de arquivos permitidos (label, extensão).

    Returns
    -------
    str or None
        Caminho do arquivo selecionado ou None se cancelado.
    """

    if tipos is None:
        tipos = [
            ("Todos os arquivos", "*.*"),
            ("Excel", "*.xlsx *.xls"),
            ("CSV", "*.csv"),
            ("JSON", "*.json"),
            ("SQL", "*.sql")
        ]

    root = Tk()
    root.withdraw()  # esconde janela principal
    root.attributes("-topmost", True)  # força ficar na frente

    try:
        caminho = filedialog.askopenfilename(
            title=titulo,
            filetypes=tipos
        )
    finally:
        root.destroy()  # garante fechamento

    if not caminho:
        return None

    return caminho
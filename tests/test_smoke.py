import importlib


def test_rag_lab_package_is_importable() -> None:
    module = importlib.import_module("rag_lab")
    assert module.__name__ == "rag_lab"

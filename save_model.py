import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os



def save_model_artifacts(model, variant, W1, b1, W2, b2, W3=None, b3=None, loss=None, epochs=None, image_count=None):    
    path = Path(__file__).resolve().parent
    path = path.joinpath(model)
    path = path.joinpath("variants")
    path = path.joinpath(variant)
    path.mkdir(parents=True, exist_ok=True)
    path.joinpath("info.txt").write_text(
        f"Feige 1.0 Modell-Info\n"
        f"--------------------\n"
        f"Loss: {loss:.6f}\n"
        f"Eingespeiste Bilder: {image_count}\n"
        f"Epochen: {epochs}\n"
        f"Hiden-Layer: 1\n"
        f"Hidden-Layer-Neurons: 128\n"
        f"Aktivierungsfunktion: ReLU\n",
        encoding="utf-8",
    )
    path = path.joinpath(f"{model}.npz")
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, loss=loss, epochs=epochs, image_count=image_count)


    print(f"Modell gespeichert in: {path}")
    return path


def resolve_model_directory(output_model_path):
    model_path = Path(output_model_path)
    if model_path.suffix.lower() == ".npz":
        return model_path.with_suffix("")
    return model_path


def save_model_artifacts(model_dir, W1, b1, W2, b2, loss, epochs, image_count):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / f"{MODEL_NAME}.npz"
    np.savez(model_file, W1=W1, b1=b1, W2=W2, b2=b2)

    info_file = model_dir / "info.txt"
    info_text = (
        "Feige 1.0 Modell-Info\n"
        "--------------------\n"
        f"Loss: {loss:.6f}\n"
        f"Eingespeiste Bilder: {image_count}\n"
        f"Epochen: {epochs}\n"
        "Hiden-Layer: 1\n"
        "Hidden-Layer-Neurons: 128\n"
        "Aktivierungsfunktion: ReLU\n"
    )
    info_file.write_text(info_text, encoding="utf-8")

    print(f"Modell gespeichert in: {model_dir}")
    return model_file


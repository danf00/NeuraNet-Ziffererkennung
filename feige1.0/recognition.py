# ! 
# ACHTUNG: Dieses Skript wurde nicht von mir selbst erstellt, es hat auch keinen neuen wirklichen Ki Aspekt es dient nur der
# Visualisierung der Ergebnisse des trainierten Modells. Es ist ein reines Pygame Frontend für das Modell.

"""
Ziffern-Erkennung mit Pygame – Zeichenfläche für ein selbstgebautes,
reines NumPy-MLP (784 -> 128 -> 10, ReLU + Softmax).

Aufruf:
    python recognition.py cnn_model_feige1.0.npz

Erwartetes .npz-Format (per numpy.savez gespeichert):
    W1 (784, 128), b1 (1, 128), W2 (128, 10), b2 (1, 10)

Steuerung:
    - Linke Maustaste gedrückt halten und über das Raster ziehen -> zeichnen
    - "Reset"-Button -> Raster leeren
    - "Classify"-Button -> Vorhersage des Modells anzeigen
"""

import sys
import numpy as np
import pygame

# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------
GRID_SIZE = 28          # 28x28 Pixel, wie MNIST
CELL_SIZE = 20           # Anzeige-Größe jeder Zelle in Pixeln
GRID_PIXELS = GRID_SIZE * CELL_SIZE

MARGIN = 30
BUTTON_WIDTH = 140
BUTTON_HEIGHT = 50
BUTTON_GAP = 30

WIDTH = GRID_PIXELS + 2 * MARGIN
HEIGHT = GRID_PIXELS + 2 * MARGIN + BUTTON_HEIGHT + MARGIN

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
DARK_GRAY = (60, 60, 60)

BRUSH_RADIUS = 1  # wie viele Nachbarzellen beim Zeichnen mit eingefärbt werden


def load_model(path: str):
    """Lädt die Gewichte eines selbstgebauten 2-Schichten-MLP aus einer .npz-Datei.
    Erwartete Arrays: W1 (784, 128), b1 (1, 128), W2 (128, 10), b2 (1, 10)."""
    print(f"Lade Modell: {path} ...")
    data = np.load(path)
    weights = {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
    }
    print("Modell geladen.")
    return weights


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # numerisch stabil: größten Wert abziehen, bevor exp() gerechnet wird
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward_pass(weights, x):
    """x: (1, 784) -> gibt (1, 10) Wahrscheinlichkeiten zurück."""
    z1 = x @ weights["W1"] + weights["b1"]
    a1 = relu(z1)
    z2 = a1 @ weights["W2"] + weights["b2"]
    a2 = softmax(z2)
    return a2


def draw_grid(screen, grid, offset_x, offset_y):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = grid[row, col]  # 0.0 (weiß) bis 1.0 (schwarz)
            shade = int(255 - value * 255)
            color = (shade, shade, shade)
            rect = pygame.Rect(
                offset_x + col * CELL_SIZE,
                offset_y + row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)  # dünne Gitterlinie


def draw_button(screen, font, rect, text):
    pygame.draw.rect(screen, WHITE, rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    label = font.render(text, True, BLACK)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)


def paint_at(grid, row, col, intensity=1.0):
    """Malt eine Zelle und lässt die Farbe sanft in die Nachbarzellen auslaufen
    (ähnlich einem weichen Pinsel), damit es eher wie eine echte Handschrift aussieht."""
    for dr in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
        for dc in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            r, c = row + dr, col + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                dist = (dr ** 2 + dc ** 2) ** 0.5
                falloff = max(0.0, 1.0 - dist * 0.6)
                grid[r, c] = min(1.0, grid[r, c] + intensity * falloff)


def classify(weights, grid):
    # Modell erwartet einen flachen Vektor (1, 784), normalisiert auf [0, 1]
    x = grid.reshape(1, GRID_SIZE * GRID_SIZE).astype("float64")
    prediction = forward_pass(weights, x)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return digit, confidence


def main():
    if len(sys.argv) != 2:
        print("Nutzung: python recognition.py cnn_model_feige1.0.npz")
        sys.exit(1)

    weights = load_model(f"variants/{sys.argv[1]}/feige1.0.npz")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ziffern-Erkennung")
    font = pygame.font.SysFont("arial", 28)
    result_font = pygame.font.SysFont("arial", 22, bold=True)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype="float32")

    grid_offset_x = MARGIN
    grid_offset_y = MARGIN

    button_y = MARGIN + GRID_PIXELS + MARGIN
    total_buttons_width = 2 * BUTTON_WIDTH + BUTTON_GAP
    buttons_start_x = (WIDTH - total_buttons_width) // 2

    reset_rect = pygame.Rect(buttons_start_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)
    classify_rect = pygame.Rect(
        buttons_start_x + BUTTON_WIDTH + BUTTON_GAP, button_y, BUTTON_WIDTH, BUTTON_HEIGHT
    )

    result_text = ""
    drawing = False

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if reset_rect.collidepoint(mx, my):
                    grid[:] = 0.0
                    result_text = ""
                elif classify_rect.collidepoint(mx, my):
                    digit, confidence = classify(weights, grid)
                    result_text = f"Erkannt: {digit}  ({confidence * 100:.1f}%)"
                else:
                    drawing = True

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False

        if drawing:
            mx, my = pygame.mouse.get_pos()
            col = (mx - grid_offset_x) // CELL_SIZE
            row = (my - grid_offset_y) // CELL_SIZE
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                paint_at(grid, row, col)

        screen.fill(BLACK)
        draw_grid(screen, grid, grid_offset_x, grid_offset_y)
        draw_button(screen, font, reset_rect, "Reset")
        draw_button(screen, font, classify_rect, "Classify")

        if result_text:
            label = result_font.render(result_text, True, WHITE)
            label_rect = label.get_rect(
                center=(WIDTH // 2, button_y + BUTTON_HEIGHT + 20)
            )
            # Fenster ggf. vergrößern, falls Text nicht reinpasst, hier einfach clippen
            screen.blit(label, label_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
# symbols.py
import pygame
import os

class SymbolLibrary:
    def __init__(self, folder="images"):
        """
        Charge les symboles depuis des fichiers PNG individuels dans le dossier donné.
        """
        self.icons = {}
        self._load_icons(folder)

    def _load_icons(self, folder):
        # mapping entre noms et fichiers
        files = {
            "archway":    "archway.png",
            "portcullis": "portcullis.png",
            "door":       "door.png",
            "locked":     "locked.png",
            "trapped":    "trapped.png",
            "secret":     "secret.png",
            "stairs_up":  "up.png",
            "stairs_down":"down.png",
        }

        for name, filename in files.items():
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                self.icons[name] = pygame.image.load(path).convert_alpha()
            else:
                print(f"[Warning] symbole manquant : {path}")

    def get(self, name):
        """Retourne la Surface pygame du symbole"""
        return self.icons.get(name)

    def draw(self, surface, name, x, y, scale=1.0, angle=0.0):
        """
        Dessine un symbole centr? sur (x, y).
        scale permet de redimensionner (0.5 = moiti? taille).
        angle (en degr?s) permet de faire pivoter l'ic?ne si n?cessaire.
        """
        icon = self.get(name)
        if not icon:
            return

        image = icon
        if scale != 1.0:
            w = max(1, int(round(icon.get_width() * scale)))
            h = max(1, int(round(icon.get_height() * scale)))
            image = pygame.transform.smoothscale(image, (w, h))
        if angle:
            image = pygame.transform.rotate(image, angle)

        rect = image.get_rect()
        rect.center = (int(round(x)), int(round(y)))
        surface.blit(image, rect.topleft)

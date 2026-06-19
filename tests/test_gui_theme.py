import unittest

from polarization_app.gui.theme import MIN_TEXT_CONTRAST, THEMES, contrast_ratio


class GuiThemeTestCase(unittest.TestCase):
    def test_theme_palettes_keep_readable_contrast(self):
        for theme in THEMES.values():
            with self.subTest(theme=theme.name):
                pairs = {
                    "text/background": (theme.text, theme.background),
                    "text/surface": (theme.text, theme.surface),
                    "input": (theme.input_text, theme.input_background),
                    "muted/background": (theme.muted, theme.background),
                    "plot": (theme.text, theme.plot_background),
                    "accent": (theme.on_accent, theme.accent),
                    "error/background": (theme.error, theme.background),
                    "warning/background": (theme.warning, theme.background),
                    "success/background": (theme.success, theme.background),
                }
                for label, (foreground, background) in pairs.items():
                    with self.subTest(pair=label):
                        self.assertGreaterEqual(contrast_ratio(foreground, background), MIN_TEXT_CONTRAST)


if __name__ == "__main__":
    unittest.main()

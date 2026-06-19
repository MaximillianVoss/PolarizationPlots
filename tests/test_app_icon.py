import unittest

from PIL import Image

from polarization_app.gui.app import APP_ICON_ICO, APP_ICON_PNG


class AppIconTestCase(unittest.TestCase):
    def test_icon_assets_exist_and_have_expected_sizes(self):
        self.assertTrue(APP_ICON_PNG.exists())
        self.assertTrue(APP_ICON_ICO.exists())

        with Image.open(APP_ICON_PNG) as png:
            self.assertEqual(png.size, (256, 256))

        with Image.open(APP_ICON_ICO) as ico:
            self.assertEqual(
                ico.ico.sizes(),
                {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)},
            )


if __name__ == "__main__":
    unittest.main()

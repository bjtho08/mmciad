import unittest
import os
from glob import glob
import sys
from PIL import Image
import numpy as np
from skimage.io import imread
import tifffile
from tensorflow.keras.utils import to_categorical
from tf_mmciad.utils.io import ImageTiler, ImageAssembler


class TestImageTiler(unittest.TestCase):
    """Test the ImageTiler class
    """

    def setUp(self):
        self.img_name = "train-3412 5V AAA WEIGERT - 2020-02-27 18.58.38{}"
        self.base_path = os.path.join("/nb_projects", "AAA_ml", "data", "WSI")
        self.raw = os.path.join(self.base_path, self.img_name.format(".tif"))
        self.g_t = os.path.join(self.base_path, "gt", self.img_name.format(".png"))
        self.tile_path = os.path.join("/home", "bthorsted", "Pictures", "tile_test")
        return super().setUp()

    def test_init_class(self):
        """Test initialization
        """
        image_tiler = ImageTiler(self.raw, (1024, 1024), self.tile_path + "/raw")
        self.assertEqual(image_tiler.img_width, 18432)
        self.assertEqual(image_tiler.img_height, 14336)
        self.assertEqual(len(image_tiler._tile_list), 285)

    def test_force_extension(self):
        """Test forcing a chosen extension rather than defaulting
        to input extension
        """
        force_extension_path = os.path.join(self.tile_path, "force_extension")
        os.makedirs(force_extension_path, exist_ok=True)
        gt_tiler = ImageTiler(
            self.g_t, (1024, 1024), force_extension_path, force_extension="tif"
        )
        self.assertEqual(os.path.splitext(gt_tiler._tile_list[0])[-1], ".tif")

    def test_get_tile(self):
        """ Test whether tiles are accessed correctly
        """
        image_tiler = ImageTiler(self.raw, (1024, 1024), self.tile_path)
        tile = image_tiler[0]
        tile1 = image_tiler[-1]
        self.assertEqual(len(tile), 2)
        self.assertTupleEqual(
            tile,
            (
                os.path.join(
                    self.tile_path,
                    self.img_name.format(""),
                    "train-3412 5V AAA WEIGERT - 2020-02-27 18.58.38_0000.tif",
                ),
                (0, 0, 1024, 1024),
            ),
        )
        self.assertNotEqual(tile, tile1)


class TestImageAssembler(unittest.TestCase):
    """Test the ImageAssembler class
    """

    def setUp(self):
        self.img_name = "train-3412 5V AAA WEIGERT - 2020-02-27 18.58.38{}"
        self.base_path = os.path.join("/nb_projects", "AAA_ml", "data", "WSI")
        self.raw = os.path.join(self.base_path, self.img_name.format(".tif"))
        self.g_t = os.path.join(self.base_path, "gt", self.img_name.format(".png"))
        self.tile_path = os.path.join("/home", "bthorsted", "Pictures", "tile_test")
        self.colors = np.array(
            [[0, 0, 0], [180, 180, 180], [0, 0, 255], [0, 255, 0], [255, 0, 0]]
        )
        self.image_tiler = ImageTiler(self.raw, (1024, 1024), self.tile_path + "/raw")
        self.image_tiler.dump(os.path.join(self.tile_path, "tiledump.yaml"))
        gt_tile_path = self.tile_path + "/gt"
        self.gt_tiler = ImageTiler(self.g_t, (1024, 1024), gt_tile_path)
        if not glob(
            os.path.join(gt_tile_path, self.img_name.format("") + "*.tif")
        ):
            for tile_path in glob(
                os.path.join(gt_tile_path, self.img_name.format("") + "*.png")
            ):
                tile = imread(tile_path)
                sparse_tile = np.zeros(shape=(*tile.shape[:2], 1), dtype="uint8")
                for cls_, color in enumerate(self.colors):
                    sparse_tile += np.expand_dims(
                        np.logical_and.reduce(tile == color, axis=-1) * cls_,
                        axis=-1,
                    ).astype("uint8")
                cat_tile = to_categorical(sparse_tile, num_classes=5)
                tifffile.imwrite(tile_path.replace("png", "tif"), np.moveaxis(cat_tile, -1, 0), imagej=True)
        self.gt_tiler.dump(os.path.join(self.tile_path, "gtdump.yaml"))
        return super().setUp()

    def test_assemble(self):
        """ Test assembling of a new image from tiles
        """
        raw_assembler = ImageAssembler(
            self.tile_path + "/raw_output_test.tif",
            os.path.join(self.tile_path, "tiledump.yaml"),
        )
        self.assertEqual(
            list(raw_assembler.tile_coords.keys())[0],
            os.path.join(
                "/home",
                "bthorsted",
                "Pictures",
                "tile_test",
                " ".join(
                    ["train-3412", "5V", "AAA", "WEIGERT", "-", "2020-02-27 18.58.38"]
                ),
                "train-3412 5V AAA WEIGERT - 2020-02-27 18.58.38_0000.tif",
            ),
        )
        raw_assembler.merge(colors=self.colors, format="PNG", mode="RBG")
        gt_assembler = ImageAssembler(
            self.tile_path + "/gt/gt_output_test.png",
            os.path.join(self.tile_path, "gtdump.yaml"),
        )
        gt_assembler.merge(colors=self.colors, format="PNG", mode="P")

    def test_assemble_from_instance(self):
        """ Test assembling of a new image from tiles by supplying an
        Imagetiler instance as input parameter
        """
        raw_assembler = ImageAssembler(
            self.tile_path + "/raw_output_test.tif", self.image_tiler
        )
        self.assertEqual(
            list(raw_assembler.tile_coords.keys())[0],
            os.path.join(
                "/home",
                "bthorsted",
                "Pictures",
                "tile_test",
                " ".join(
                    ["train-3412", "5V", "AAA", "WEIGERT", "-", "2020-02-27 18.58.38"]
                ),
                "train-3412 5V AAA WEIGERT - 2020-02-27 18.58.38_0000.tif",
            ),
        )
        raw_assembler.merge(colors=self.colors, format="PNG", mode="RBG")
        gt_assembler = ImageAssembler(
            self.tile_path + "/gt/gt_output_test.png", self.gt_tiler
        )
        gt_assembler.merge(colors=self.colors, format="PNG", mode="P")

    def test_dev(self):
        """ Test the multi-channel merger
        """
        gt_assembler = ImageAssembler(
            self.tile_path + "/gt/gt_output_test.tif", self.gt_tiler, ".tif"
        )
        img_list = gt_assembler.merge_multichannel()
        img = Image.open(img_list[0])
        print(img.width, img.height, file=sys.stderr)
        self.assertEqual(img.width, 18432)
        self.assertEqual(img.height, 14336)
        self.assertEqual(len(img_list), 5)

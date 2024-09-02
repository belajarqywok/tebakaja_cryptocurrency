import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        # Model Builder
        'model_builder.pyx',

        # Data Processor
        'data_processor.pyx',

        # Post Processor
        'post_processor.pyx',

        # Trainer
        'trainer.pyx',

        # Main
        'main.pyx'
    ]),

    include_dirs = [ np.get_include() ]
)

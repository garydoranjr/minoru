#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy as np

c_minoru = Extension(
    'c_minoru',
    sources=['c_minoru/pylibcam.cpp',
             'c_minoru/libcam.cpp' ],
    include_dirs=['c_minoru', np.get_include()]
)

setup(
    name='minoru',
    version='1.0',
    description=(
      "An Python interface to the Minoru stereo camera."
    ),
    author='Gary Doran',
    author_email='garydoranjr@gmail.com',
    url='https://github.com/garydoranjr/minoru.git',
    license="GNU GPL (see the LICENSE file)",
    platforms=['unix'],
    packages=['minoru'],
    ext_modules=[c_minoru]
)

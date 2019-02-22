from distutils.core import setup
from Cython.Build import cythonize


setup(
    name='UGR_ImageFeatureExtraction',
    version='',
    packages=[''],
    url='',
    license='',
    author='jisern',
    author_email='',
    description=''
)

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
setup(
    ext_modules = cythonize("src/_functions.pyx", **ext_options)
)

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

# ext_modules = [Extension("hello", ["hello.pyx"])]

# setup(
#   name = 'Hello world app',
#   cmdclass = {'build_ext': build_ext},
#   ext_modules = ext_modules
# )

# from distutils.core import setup
# from Cython.Build import cythonize

# setup(name="thread_demo", ext_modules=cythonize('thread_demo.pyx'),)

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

# ext_modules=[ Extension("fastloop",
#               ["fastloop.pyx"],
#               libraries=["m"],
#               extra_compile_args = ["-ffast-math"])]

# setup(
#   name = "fastloop",
#   cmdclass = {"build_ext": build_ext},
#   ext_modules = ext_modules)

#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(name="fastloop", ext_modules=cythonize('fastloop.pyx'),)

from distutils.core import setup
#from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import cython_gsl

ext_modules=[ Extension("fastloop",
                        ["fastloop.pyx"],
                        libraries=["m"],
                        extra_compile_args = ["-ffast-math"])]

setup(
      name = "fastloop",
      include_dirs = [cython_gsl.get_include()],
      cmdclass = {"build_ext": build_ext},
      ext_modules = [Extension("my_cython_script",
                             ["src/my_cython_script.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()])]
    )

#ext_modules=[
#    Extension("thread_demo",
#              ["thread_demo.pyx"],
#               libraries=["m"],
#               extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#               extra_link_args=['-fopenmp']
#               ) 
# ]
#
#setup(
#   name = "thread_demo",
#   cmdclass = {"build_ext": build_ext},
#   ext_modules = ext_modules
# )
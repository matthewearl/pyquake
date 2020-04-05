#!/usr/bin/env python

from distutils.core import setup


setup(name='Pyquake',
      version='1.0',
      entry_points={
          'console_scripts': [
                'demo_viewer = pyquake.render:demo_viewer_main',
                'ray_tracer = pyquake.ray:raytracer_main',
                'pyqclient = pyquake.client:client_main',
                'aiopyqclient = pyquake.client:aioclient_main',
                'aiodgram = pyquake.aiodgram:main',
                'demo_parser = pyquake.proto:demo_parser_main',
                'pyq_monitor_demos = pyquake.demo:monitor_demos',
          ]
      },
      description='Python Quake client',
      install_requires=['numpy', 'scipy'],
      author='Matt Earl',
      packages=['pyquake'])


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
                'pyq_extract_lights = pyquake.mapsource:extract_lights_main',
                'pyq_pak_extract = pyquake.pak:pak_extract_main',
                'demo_stats = pyquake.demstats:demo_stats_entrypoint',
                'pyq_dump_progs = pyquake.progs:dump_progs_main',
          ]
      },
      description='Python Quake client',
      install_requires=['numpy', 'scipy', 'parsley'],
      author='Matt Earl',
      packages=['pyquake'])


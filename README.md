[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

# PyQuake

Various parts of Quake 1 ported into Python.  Some interesting modules:

- `pyquake.proto`:  Parsing code for the quake network (and demo) protocol.
- `pyquake.client`:  A programmatic asyncio Quake client.
- `pyquake.blend{demo,mdl,bsp}`:  Load various quake structures into blender.
- `pyquake.pak`:  Mapping interface to the Quake filesystem.

Also the following entry points are provided:

- `demo_viewer`:  View demo(s) in an OpenGL viewer.
- `demo_parser`:  Parse demos into a human readable form.
- `pyq_pak_extract`: View and extract pak files.


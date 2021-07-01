[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

# PyQuake

Various parts of Quake 1 ported into Python.  Some interesting modules:

- `pyquake.proto`:  Parsing code for the quake network (and demo) protocol.
- `pyquake.client`:  A programmatic asyncio Quake client.
- `pyquake.blend{demo,mdl,bsp}`:  Load various quake structures into Blender.  If you want to try this out see [these instructions](blender.md).  Also see my [blog post](http://matthewearl.github.io/2021/06/20/quake-blender/) describing how this works.
- `pyquake.pak`:  Mapping interface to the Quake filesystem.

Also the following entry points are provided:

- `demo_viewer`:  View demo(s) in an OpenGL viewer.
- `demo_parser`:  Parse demos into a human readable form.
- `pyq_pak_extract`: View and extract pak files.


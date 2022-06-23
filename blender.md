# Demo to Blender scene converter

This repo contains utilities for converting Quake demo files (.dem) into Blender
scenes.  Tested with Blender 3.2.0

I wrote a [blog post](http://matthewearl.github.io/2021/06/20/quake-blender/)
describing at a high level how this works.  Here's a video showing the output:

[![Quake path traced in Blender](https://img.youtube.com/vi/uX0Ye7qhRd4/0.jpg)](https://www.youtube.com/watch?v=uX0Ye7qhRd4) 


## Setup

Setup is complicated by the fact that we need to be able to import external
libraries into Blender's Python path. To do this you'll need a Python virtualenv
installed on your machine.  The Python binary that the virtualenv uses needs to
be the same minor version as Blender's (ie. 3.10.x if using Blender 3.2.0).

On Linux one way to achieve this is by using
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) on the
Python binary distributed with Blender:

```
mkvirtualenv blendquake -p ~/blender-3.2.0-linux-x64/3.2/python/bin/python3.10
```

Once activated, checkout this repo, `cd` into it, and `pip install -e . -v` to
install the repo in development mode

Next, start Blender.  Set the render engine to cycles. From an empty scene, open
the text editor, create a new script, and paste in the following code:

```
# Setup path.  Change the paths here to point to your virtualenv's
# site-packages, and your local pyquake install.
from os.path import expanduser
import sys

for x in [
     expanduser('~/.virtualenvs/blendquake/lib/python3.7/site-packages'),
     expanduser('~/pyquake'),
      ]:
    if x not in sys.path:
        sys.path.append(x)


# Import everything, and reload modules that we're likely to change>
import importlib
import logging
import json
import pyquake.blenddemo
import pyquake.blendmdl
import pyquake.blendbsp
from pyquake import pak

importlib.reload(pyquake.blendbsp)
importlib.reload(pyquake.blendmdl)
blenddemo = importlib.reload(pyquake.blenddemo)

# Log messages should appear on the terminal running Blender
logging.getLogger().setLevel(logging.INFO)

# Settings for setting texture brightnesses, etc.
with open(expanduser('~/pyquake/config.json')) as f:
  config = json.load(f)

# Directory containing `id/`
fs = pak.Filesystem(expanduser('~/.quakespasm/'))

# Demo file to load
demo_fname = expanduser('~/Downloads/e1m6_conny.dem')

with open(demo_fname, 'rb') as demo_file:
    world_obj, obj_mgr = blenddemo.add_demo(demo_file, fs, config, fps=360)

world_obj.scale = (0.01,) * 3

```

Change paths to point to the correct places, and execute.  Log messages
should start appearing on the terminal and after a few seconds the loaded demo
should appear.

The above produces a root `demo` empty, under which the various other entities
appear.

It's recommended to save the .blend file just before executing so you can
quickly revert and re-run the script if necessary.


## Activating the demo camera

Click on the camera icon next to the demo_cam object in the outliner to make the
current camera the first person view.

You can use other cameras, however this may introduce extra noise since
emitters' sample_as_light (aka Multiple Importance Sampling) property is
keyframed according to the player's position, to avoid sampling occluded lights.


## Setting the correct framerate

Blender only allows keyframes to be set on integer frame numbers.  Quake
typically operates at 72 frames per second, so to avoid jerky motion the
framerate needs to be a multiple of 72.  If you want an output framerate of
60fps, select a custom framerate of 360 in the Dimensions section of Output
Properties, and then set the Step setting to 6.   360 is chosen since it's the
lowest common multiple of 72 and 60, and 6 is chosen since 360 divided by 6
gives the desired framerate.


## Config file

The `config.json` referred to above has a number of settings that can be
changed:

- `models.<name>.force_fullbright`: Make the whole model an emitter,
  otherwise just those parts of the texture that are in the fullbright palette
  will be.
- `models.<name>.strength`: Brightness of the emitter.
- `models.<name>.sample_as_light`: Whether to set the `sample_as_light` (aka
  multiple importance sampling) flag on the material.
- `models.<name>.bbox`: A pair of (mins, maxs) 3-tuples describing the range of
  influence of the light emitted by this object.  Required if `sample_as_light`
  is true.
- `models.<name>.no_naim`: Don't animate the model.  Currently used for flame
  models since the movement introduces temporal inconsistency in the noise.
- `maps.<name>.fullbright_object_overlay`: Separate out fullbright regions of
  textures with the `overlay` flag set into their own objects.  This is to make
  multiple importance sampling more efficient, since a large object with a small
  fullbright area will be sampled uniformly across the object.
- `maps.<name>.textures.<name>.strength`:  Emission strength for fullbright
  parts of the texture.
- `maps.<name>.textures.<name>.tint`:  Emitted colours are multiplied by this
  4-vector.  Use to tune light colour.
- `maps.<name>.textures.<name>.overlay`: Enable / disable fullbright object
  overlay for this texture.  See `maps.<name>.fullbright_object_overlay` for
  details.
- `maps.<name>.textures.<name>.sample_as_light`: Whether to set the
  `sample_as_light` (aka multiple importance sampling) flag on the material.
- `maps.<name>.texture.<name>.bbox`: A pair of (mins, maxs) 3-tuples describing
  the range of influence of the light emitted by this surface.  Required if
  `sample_as_light` is true.
- `maps.<name>.lights.*`: See below.

The `sample_as_light` flag for model and (static) object materials is set to
False whenever the demo_cam is out of range of the light, according to the map's
PVS data, and the view frustum.  This is to avoid wasting samples on occluded
lights.


## Adding lights from .map files

These scripts primarily illuminate the map by treating fullbright textures as
emissive materials.  However, this approach can still leave dark areas.  Quake
maps are normally illuminated by light entities, unfortunately these are lost
when the map is compiled.  To recover them run the script `pyq_extract_lights`
on the relevant map source file.  Original id map sources are available from
[here](https://rome.ro/news/2016/2/14/quake-map-sources-released):

```
pyq_extract_lights pyq_extract_lights ~/quake-map-sources/E1M6.MAP > \
    ~/quake-map-sources/e1m6-lights.json
```

The above script produces some JSON which can be parsed and patched into the
`lights` section of the map config:

```
with open(expanduser('~/pyquake/config.json')) as f:
    config = json.load(f)

with open(expanduser('~/quake-map-sources/e1m6-lights.json')) as f:
    config['maps']['e1m6']['lights'].update(json.load(f))
```

This will likely put way too many lights in the scene for two reasons:

- Some lights will double up emissive materials that we already produce.
- Quake's lightmap generation only uses direct illumination, therefore level
  designers inserted secondary lights to simulate bounced reflections.

The recommonended workflow is to select just a few of these lights such that the
lighting resembles that of the original game (albeit more realistically).  To do
this, select the lights you wish to keep and take a note of their object names.
Finally, copy the lights from the lights JSON file into the relevant section of
the global `config.json` file.  You may need to tweak the brightness to achieve
the original effect.  Beware that adding too many lights can introduce noise.


## Extracting pak files

Marathon runs are typically archived inside `.pak` files.  Since `add_demo`
requires a demo file, you'll need a way to extract the `.pak`:

```
$ pyq_pak_extract -h
usage: pyq_pak_extract [-h] [-l] [-x] pak-file-name [target-dir]

Extract / list pak archives

positional arguments:
  pak-file-name
  target-dir

optional arguments:
  -h, --help     show this help message and exit
  -l, --list     list archive contents
  -x, --extract  extract archive contents
```


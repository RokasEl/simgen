# Using `simgen` with ZnDraw

## Web app

This section describes how to generate structures using the online web app, hosted [here](https://zndraw.icp.uni-stuttgart.de/). If something breaks, clicking EXIT and reloading the page should fix it. See the in-app help (`i` button in the top bar) for general ZnDraw controls.

### Generation menu

To access the generation menu, click 'Interaction' in the left sidebar and select 'DiffusionModelling' from the 'Method' dropdown menu.

The modifier has three run types:

- **Generate**: Generates structures using similarity kernels.
- **Hydrogenate**: Adds hydrogen atoms to the active structure.
- **Relax**: Uses a pretrained MACE force field to relax/optimize the active structure after hydrogenation.

Clicking "Run modifier" will perform the selected action on the active structure. In addition, a clickable bookmark is created at the start of each run, allowing you to easily revert to the original structure.

#### Generation Settings

- **Num Steps**: Controls the number of integrator steps during generation. 50 is a good starting point.
- **Atom Number**: Can be static or determined dynamically based on the length of the guiding curve (see [Guided Generation](#guided-generation)). Select 'PerAngstrom' for dynamic atom number.
- **Guiding Force Multiplier**: Adjusts how strongly atoms are attracted to the guiding point cloud. Default of 1 works for simple tasks. Increase if the generated structure is fragmented.

#### Hydrogenate and Relax Settings

These modes have just one parameter controlling the number of relaxation steps. Defaults are suitable for most uses.

### Guided Generation

We can create structures with arbitrary shapes by providing a point cloud prior. To draw a guiding curve:

1. Press `x` or the 'Draw' button to enter drawing mode.
2. **Click on atoms** to place guiding points.
3. Press `x` or the 'Draw' button to exit drawing mode.

A smooth curve that passes through all the guiding points is automatically generated.

The guiding curve is easy to modify:

 - Clicking a point selects it:
     - Move the selected point with the on-screen controls.
     - Delete the selected point by clicking `Backspace`.
 - Clicking the blue midpoint of a segment inserts a new point.

To place points in free 3D space:

1. Create a canvas using the Drawing option in the left sidebar.
2. Now in drawing mode, points will snap to the canvas.

To modify the canvas:

- Press f to enter canvas mode. Now you can rotate and translate the atoms while the canvas stays fixed.
- Hold shift and scroll to resize the canvas and access on-screen controls.

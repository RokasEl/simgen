# Using `moldiff` with ZnDraw

## Web app

This section describes how to generate structures using the online web app, hosted [here](https://zndraw.pythonf.de/). If something breaks, clicking EXIT and reloading the page should fix it. See the in-app help (`i` button in the top bar) for general ZnDraw controls.

### Generation menu

To access the generation menu, click 'Interaction' in the left sidebar and select 'DiffusionModelling' from the 'Method' dropdown menu.

The modifier has three run types:

- **Generate**: The primary function for structure generation using similarity kernels.
- **Hydrogenate**: Adds hydrogen atoms to the active structure.
- **Relax**: Uses a pretrained MACE force field to relax/optimize the active structure after hydrogenation.

Clicking "Run modifier" will perform the selected action on the active structure.

#### Generation Settings

- **Num Steps**: Controls diffusive steps taken during generation. Higher values result in slower generation. 50 is a good starting point.
- **Atom Number**: Can be static or determined dynamically based on the length of the guiding curve (see [Guided Generation](#guided-generation)). Select 'PerAngstrom' for dynamic atom number.
- **Guiding Force Multiplier**: Adjusts how strongly atoms are attracted to the guiding point cloud. Default of 1 works for simple tasks. Increase for more complicated guiding curves.

#### Hydrogenate and Relax Settings

These modes have just one parameter controlling the number of relaxation steps. Defaults are suitable for most uses.

### Guided Generation

We can create structures with arbitrary shapes by providing a point cloud prior. To draw a guiding curve:

1. Press `x` or the 'Draw' button to enter drawing mode.
2. Click atoms to place guiding points.
3. Create a curve by adding multiple points.
4. Press `x` or the 'Draw' button to exit drawing mode.

To modify the guiding curve, click a point to select it. Then you can move it with the on-screen controls. When a point is selected:

- `Backspace` deletes the point.
- `d` duplicates the point, inserting it between current and previous.

To place free points in 3D space:

1. Create a canvas using the Drawing option in the left sidebar.
2. Now in drawing mode, points will snap to the canvas.

To modify the canvas:

- Press f to enter canvas mode, where the canvas stays fixed as you move atoms with the mouse.
- Hold shift and scroll to resize the canvas and access movement controls.

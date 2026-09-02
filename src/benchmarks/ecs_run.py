from ecs.core.models import SIREpidemicModel
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

model = SIREpidemicModel(seed = 28022026, enable_quarantine = False,
                         initial_infected = 20, world_size= 500,
                         beta_spatial = 0.3, beta_network = 0.25,
                         spatial_new = True, network_new = True,
                         space_attribute_similarity = True, n_agents = 1000, dt = 1.0,
                         dispersion = 0.65) 

model.run(max_steps=400)

time_series = model._collect_data()
spatial_location_series = model.get_spatial_data()

#print(time_series.items())

#print(spatial_location_series.items())

def plot_epidemic_curves(time_series, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(time_series["susceptible"], label="Susceptible")
    plt.plot(time_series["infected"], label="Infected")
    plt.plot(time_series["recovered"], label="Recovered")
    plt.plot(time_series["death"], label="Death")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Individuals")
    plt.title("Epidemic Curves")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved epidemic curves to {save_path}")
    plt.show()
    
def plot_spatial_snapshot(spatial_location_series, time_step=None, save_path=None):
    colors = {
        "susceptible": "blue",
        "infected": "red",
        "recovered": "green",
        "death": "black"
    }

    indexed_data = defaultdict(lambda: defaultdict(list))
    all_x, all_y = [], []
    max_t = 0

    for status, data_list in spatial_location_series.items():
        for t, x, y in data_list:
            t_int = round(int(t))  # safer than bare int() for floats
            indexed_data[t_int][status].append((x, y))
            all_x.append(x)
            all_y.append(y)
            if t_int > max_t:
                max_t = t_int

    if not all_x:
        print("No data found.")
        return

    # Debug: verify frames have distinct data
    print(f"Total frames: {max_t + 1}")
    print(f"Frames with data: {sorted(indexed_data.keys())[:10]}...")
    sample_frames = sorted(indexed_data.keys())[:3]
    for f in sample_frames:
        counts = {s: len(v) for s, v in indexed_data[f].items()}
        print(f"  Frame {f}: {counts}")

    x_lims = (min(all_x) - 1, max(all_x) + 1)
    y_lims = (min(all_y) - 1, max(all_y) + 1)

    if time_step is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        frame_data = indexed_data[time_step]
        for status, color in colors.items():
            locs = frame_data.get(status, [])
            if locs:
                xs, ys = zip(*locs)
                ax.scatter(xs, ys, c=color, label=status, alpha=0.6, s=20)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.legend()
        ax.set_title(f"Time Step: {time_step}")
        plt.show()

    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Pre-build legend handles so legend persists after ax.clear()
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markersize=8, label=s)
            for s, c in colors.items()
        ]

        def animate(frame):
            ax.clear()
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_title(f"Time Step: {frame}")
            ax.legend(handles=legend_handles, loc='upper right')

            current_frame_data = indexed_data[frame]
            for status, color in colors.items():
                locations = current_frame_data.get(status, [])
                if locations:
                    xs, ys = zip(*locations)
                    ax.scatter(xs, ys, c=color, alpha=0.6, s=20)

        anim = FuncAnimation(
            fig, animate,
            frames=sorted(indexed_data.keys()),  # ← KEY FIX: only iterate frames that exist
            interval=100,
            blit=False,
            repeat=True
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
            plt.close(fig)  # ← prevents blank window after save
        else:
            plt.show()

def plot_spatial_heatmap(spatial_location_series, time_step=None, save_path=None,
                          world_size=500, grid_resolution=200, sigma=3.5):

    # RGBA colour per compartment (R, G, B) — alpha handled by density
    compartment_colours = {
        "susceptible": np.array([0.0,  0.4,  1.0]),   # blue
        "infected":    np.array([1.0,  0.1,  0.0]),   # red
        "recovered":   np.array([0.0,  0.9,  0.2]),   # green
        "death":       np.array([0.9,  0.9,  0.9]),   # white/grey
    }

    cell_width = world_size / grid_resolution

    # ── Index data — identical to scatter function
    indexed_data = defaultdict(lambda: defaultdict(list))
    max_t = 0

    for status, data_list in spatial_location_series.items():
        for t, x, y in data_list:
            t_int = round(int(t))
            indexed_data[t_int][status].append((x, y))
            if t_int > max_t:
                max_t = t_int

    if not indexed_data:
        print("No data found.")
        return

    print(f"Total frames: {max_t + 1}")
    print(f"Frames with data: {sorted(indexed_data.keys())[:10]}...")
    for f in sorted(indexed_data.keys())[:3]:
        counts = {s: len(v) for s, v in indexed_data[f].items()}
        print(f"  Frame {f}: {counts}")

    extent = [0, world_size, 0, world_size]

    def build_density(frame_data, status):
        grid = np.zeros((grid_resolution, grid_resolution))
        for x, y in frame_data.get(status, []):
            ci = min(int(x / cell_width), grid_resolution - 1)
            cj = min(int(y / cell_width), grid_resolution - 1)
            grid[cj, ci] += 1.0
        return gaussian_filter(grid, sigma=sigma)

    def build_composite(frame_data):
        """
        Blends all compartment densities into a single RGBA image.
        Each compartment contributes its colour weighted by its
        normalised density — brighter = more agents in that cell.
        """
        # Compute and normalise each compartment density
        densities = {}
        for comp in compartment_colours:
            d = build_density(frame_data, comp)
            d_max = d.max()
            densities[comp] = d / d_max if d_max > 0 else d

        # Build RGB image — start black
        rgb = np.zeros((grid_resolution, grid_resolution, 3))
        alpha = np.zeros((grid_resolution, grid_resolution))

        for comp, colour in compartment_colours.items():
            d = densities[comp]
            # Each compartment adds its colour scaled by its density
            rgb   += d[:, :, np.newaxis] * colour[np.newaxis, np.newaxis, :]
            alpha += d

        # Clip RGB to [0, 1] — additive blending can exceed 1 in overlap zones
        # which naturally produces brighter mixed colours (e.g. red+blue = magenta)
        rgb = np.clip(rgb, 0, 1)

        # Alpha channel — normalise so full occupancy = fully opaque
        alpha_max = alpha.max()
        alpha_norm = alpha / alpha_max if alpha_max > 0 else alpha
        alpha_norm = np.clip(alpha_norm, 0, 1)

        # Stack into RGBA
        rgba = np.dstack([rgb, alpha_norm])
        return rgba

    # ── Legend handles — fixed, shown once
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colour, markersize=10, label=comp.capitalize())
        for comp, colour in compartment_colours.items()
    ]

    # ── Single frame
    if time_step is not None:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        rgba = build_composite(indexed_data[time_step])
        ax.imshow(rgba, origin="lower", extent=extent,
                  interpolation="bilinear", aspect="auto")

        counts = {c: len(indexed_data[time_step].get(c, []))
                  for c in compartment_colours}
        ax.set_title(
            f"Step {time_step}   |   "
            f"S={counts['susceptible']}  "
            f"I={counts['infected']}  "
            f"R={counts['recovered']}  "
            f"D={counts['death']}",
            color="white", fontsize=13, pad=10
        )
        ax.tick_params(colors="white")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#222222")
        ax.legend(handles=legend_handles, loc="upper right",
                  facecolor="#111111", labelcolor="white", fontsize=10)

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, facecolor="black")
            print(f"Snapshot saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    # ── Animation
    else:
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        rgba = build_composite(indexed_data[0])
        im = ax.imshow(rgba, origin="lower", extent=extent,
                       interpolation="bilinear", aspect="auto")

        ax.tick_params(colors="white")
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#222222")
        ax.legend(handles=legend_handles, loc="upper right",
                  facecolor="#111111", labelcolor="white", fontsize=10)

        step_text = fig.text(
            0.5, 0.97, "Step 0",
            ha="center", va="top",
            color="white", fontsize=13, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        def animate(frame):
            rgba = build_composite(indexed_data[frame])
            im.set_data(rgba)

            counts = {c: len(indexed_data[frame].get(c, []))
                      for c in compartment_colours}
            step_text.set_text(
                f"Step {frame}   |   "
                f"S={counts['susceptible']}  "
                f"I={counts['infected']}  "
                f"R={counts['recovered']}  "
                f"D={counts['death']}"
            )
            return [im, step_text]

        anim = FuncAnimation(
            fig, animate,
            frames=sorted(indexed_data.keys()),
            interval=100,
            blit=True,
            repeat=True
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

def plot_spatial_3d_wave(spatial_location_series, time_step=None, save_path=None,
                          world_size=500, grid_resolution=80, sigma=5.0,
                          compartment="infected"):
    """
    Renders epidemic density as an animated 3D surface wave.

    Parameters
    ----------
    compartment : str
        Which compartment to render as the wave surface.
        Use "infected" for epidemic wave propagation.
        Pass None to composite all compartments into a single surface.
    sigma : float
        Gaussian smoothing radius in grid cells.
    grid_resolution : int
        Keep lower for 3D (80-100) — 3D rendering is more expensive than 2D.
    """

    compartment_colours = {
        "susceptible": cm.Blues,
        "infected":    cm.inferno,
        "recovered":   cm.Greens,
        "death":       cm.bone
    }

    cell_width = world_size / grid_resolution

    # ── Index data
    indexed_data = defaultdict(lambda: defaultdict(list))
    max_t = 0

    for status, data_list in spatial_location_series.items():
        for t, x, y in data_list:
            t_int = round(int(t))
            indexed_data[t_int][status].append((x, y))
            if t_int > max_t:
                max_t = t_int

    if not indexed_data:
        print("No data found.")
        return

    print(f"Total frames: {max_t + 1}")
    print(f"Frames with data: {sorted(indexed_data.keys())[:10]}...")
    for f in sorted(indexed_data.keys())[:3]:
        counts = {s: len(v) for s, v in indexed_data[f].items()}
        print(f"  Frame {f}: {counts}")

    # Grid coordinates — computed once
    x_grid = np.linspace(0, world_size, grid_resolution)
    y_grid = np.linspace(0, world_size, grid_resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)

    def build_density(frame_data, status):
        grid = np.zeros((grid_resolution, grid_resolution))
        for x, y in frame_data.get(status, []):
            ci = min(int(x / cell_width), grid_resolution - 1)
            cj = min(int(y / cell_width), grid_resolution - 1)
            grid[cj, ci] += 1.0
        return gaussian_filter(grid, sigma=sigma)

    def build_composite_density(frame_data):
        """Sum all compartment densities into one surface — shows total activity"""
        weights = {"susceptible": 0.3, "infected": 1.0,
                   "recovered": 0.5, "death": 0.1}
        combined = np.zeros((grid_resolution, grid_resolution))
        for comp, w in weights.items():
            combined += build_density(frame_data, comp) * w
        return combined

    # Determine which density function to use
    use_composite = compartment is None
    cmap = cm.inferno if use_composite else compartment_colours.get(compartment, cm.inferno)

    # Pre-compute global zmax for consistent z-axis across all frames
    zmax = 0
    for step in indexed_data:
        if use_composite:
            d = build_composite_density(indexed_data[step])
        else:
            d = build_density(indexed_data[step], compartment)
        zmax = max(zmax, d.max())
    zmax = max(zmax, 1e-6)

    # ── Single frame
    if time_step is not None:
        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_facecolor("black")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("black")

        if use_composite:
            zz = build_composite_density(indexed_data[time_step])
            title_comp = "All Compartments (weighted)"
        else:
            zz = build_density(indexed_data[time_step], compartment)
            title_comp = compartment.capitalize()

        zz_norm = zz / zmax  # normalise to [0,1] for colouring

        surf = ax.plot_surface(
            xx, yy, zz,
            facecolors=cmap(zz_norm),
            rstride=1, cstride=1,
            linewidth=0, antialiased=True,
            shade=True
        )

        counts = {c: len(indexed_data[time_step].get(c, []))
                  for c in compartment_colours}
        ax.set_title(
            f"{title_comp} — Step {time_step}   |   "
            f"S={counts['susceptible']}  I={counts['infected']}  "
            f"R={counts['recovered']}  D={counts['death']}",
            color="white", fontsize=11, pad=12
        )
        _style_3d_ax(ax, world_size, zmax)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=120, facecolor="black")
            print(f"Snapshot saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    # ── Animation
    else:
        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_facecolor("black")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("black")

        # Initial surface
        zz = build_density(indexed_data[0], compartment or "infected")
        zz_norm = zz / zmax

        # plot_surface cannot be updated in place with set_data like imshow
        # so we redraw the surface each frame — store ref to remove it
        surf_container = [ax.plot_surface(
            xx, yy, zz,
            facecolors=cmap(zz_norm),
            rstride=1, cstride=1,
            linewidth=0, antialiased=True,
            shade=True
        )]

        _style_3d_ax(ax, world_size, zmax)

        step_text = fig.text(
            0.5, 0.96, "Step 0",
            ha="center", va="top",
            color="white", fontsize=12, fontweight="bold"
        )

        def animate(frame):
            # Remove previous surface — 3D surfaces must be redrawn
            surf_container[0].remove()

            frame_data = indexed_data[frame]
            if use_composite:
                zz = build_composite_density(frame_data)
            else:
                zz = build_density(frame_data, compartment)

            zz_norm = zz / zmax
            surf_container[0] = ax.plot_surface(
                xx, yy, zz,
                facecolors=cmap(zz_norm),
                rstride=1, cstride=1,
                linewidth=0, antialiased=True,
                shade=True
            )

            counts = {c: len(frame_data.get(c, []))
                      for c in compartment_colours}
            step_text.set_text(
                f"Step {frame}   |   "
                f"S={counts['susceptible']}  "
                f"I={counts['infected']}  "
                f"R={counts['recovered']}  "
                f"D={counts['death']}"
            )
            return [surf_container[0], step_text]

        anim = FuncAnimation(
            fig, animate,
            frames=sorted(indexed_data.keys()),
            interval=150,       # slightly slower than 2D — 3D needs more render time
            blit=False,         # blit=True doesn't work reliably with 3D surfaces
            repeat=True
        )

        if save_path:
            anim.save(save_path, writer='pillow', fps=8, dpi=100)
            print(f"3D wave animation saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()


def _style_3d_ax(ax, world_size, zmax):
    """Shared 3D axis styling — dark theme"""
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_zlim(0, zmax * 1.1)
    ax.set_xlabel("X", color="white", labelpad=8)
    ax.set_ylabel("Y", color="white", labelpad=8)
    ax.set_zlabel("Density", color="white", labelpad=8)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.grid(True, color="#222222", linewidth=0.5)
    # Viewing angle — adjust for best perspective
    ax.view_init(elev=35, azim=-60)

def plot_epidemic_rings(spatial_location_series, seed_locations: list,
                         world_size=500, n_rings=20, save_path=None):
    """
    seed_locations : list of (x, y) tuples — initial infected agent positions
    Plots mean infection density as a function of distance from seeds over time
    """

    indexed_data = defaultdict(lambda: defaultdict(list))
    max_t = 0
    for status, data_list in spatial_location_series.items():
        for t, x, y in data_list:
            t_int = round(int(t))
            indexed_data[t_int][status].append((x, y))
            if t_int > max_t:
                max_t = t_int

    max_dist = world_size * np.sqrt(2)
    ring_edges = np.linspace(0, max_dist, n_rings + 1)
    ring_centers = (ring_edges[:-1] + ring_edges[1:]) / 2

    def mean_seed_distance(x, y):
        return min(
            np.sqrt((x - sx)**2 + (y - sy)**2)
            for sx, sy in seed_locations
        )

    # Sample every 5th step to keep plot readable
    steps = sorted(indexed_data.keys())[::5]
    cmap = plt.cm.plasma
    colors = [cmap(i / len(steps)) for i in range(len(steps))]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for idx, step in enumerate(steps):
        ring_counts = np.zeros(n_rings)
        for x, y in indexed_data[step].get("infected", []):
            d = mean_seed_distance(x, y)
            bin_idx = np.searchsorted(ring_edges, d, side="right") - 1
            if 0 <= bin_idx < n_rings:
                ring_counts[bin_idx] += 1
        ax.plot(ring_centers, ring_counts, color=colors[idx],
                alpha=0.7, linewidth=1.5, label=f"t={step}" if idx % 3 == 0 else "")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=steps[0], vmax=steps[-1]))
    plt.colorbar(sm, ax=ax, label="Time step").ax.yaxis.set_tick_params(color="white")
    ax.set_xlabel("Distance from seed", color="white")
    ax.set_ylabel("Infected count", color="white")
    ax.set_title("Epidemic Wave Front Propagation", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, facecolor="black")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    plot_epidemic_curves(time_series, save_path="epidemic_curves.png")
    plot_spatial_snapshot(spatial_location_series, save_path="spatial_animation.gif")
    plot_spatial_heatmap(model.spatial_location_series_data, save_path="spatial_heatmap.gif")
    plot_spatial_3d_wave(model.spatial_location_series_data,
                     save_path="wave_composite.gif",
                     world_size=500,
                     compartment=None)
    plot_epidemic_rings(spatial_location_series,
                    seed_locations=model.seed_locations,
                    world_size=model.world_size,
                    save_path="epidemic_ring_plot.png")
    
    
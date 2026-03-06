from ecs.core.models import SIREpidemicModel
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from collections import defaultdict

model = SIREpidemicModel(seed = 28022026, enable_quarantine = True,
                         initial_infected = 10, world_size= 500,
                         spatial_new = False, network_new = False,
                         space_attribute_similarity = False, n_agents = 500) #quar = 0.45 gave interesting plot

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

if __name__ == "__main__":
    plot_epidemic_curves(time_series, save_path="epidemic_curves.png")
    plot_spatial_snapshot(spatial_location_series, save_path="spatial_animation.gif")
    
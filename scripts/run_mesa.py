"""Mesa simulation entry point (same demo configuration as ``run_ecs.py``).

Run from the ``project_outputs`` folder to keep the generated files together:

    cd project_outputs
    uv run python ../scripts/run_mesa.py            # summary only
    uv run python ../scripts/run_mesa.py --plot     # save the epidemic and spatial plots
"""
import argparse
import time

from epidemic_sim.mesa import MesaSIREpidemicModel


def main():
    parser = argparse.ArgumentParser(description="Run the Mesa (OOP) epidemic model.")
    parser.add_argument("--n-agents", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=28022026)
    parser.add_argument("--plot", action="store_true", help="save epidemic and spatial plots")
    args = parser.parse_args()

    # Same parameters as the ECS demo in scripts/run_ecs.py
    model = MesaSIREpidemicModel(seed=args.seed, enable_quarantine=False,
                                 initial_infected=20, world_size=500,
                                 beta_spatial=0.3, beta_network=0.25,
                                 spatial_new=True, network_new=True,
                                 space_attribute_similarity=True,
                                 n_agents=args.n_agents, dt=1.0, dispersion=0.65)
    start = time.perf_counter()
    model.run(max_steps=args.steps)
    elapsed = time.perf_counter() - start

    series = model.time_series_data
    print(f"steps run: {len(series['time'])} | time: {elapsed:.1f}s | "
          f"peak infected: {max(series['infected'])} | "
          f"recovered: {series['recovered'][-1]} | deaths: {series['death'][-1]} | "
          f"never infected: {series['susceptible'][-1]}")

    if args.plot:
        from epidemic_sim.analysis.plotting import (
            plot_epidemic_curves,
            plot_spatial_3d_wave,
            plot_spatial_heatmap,
            plot_spatial_snapshot,
        )

        spatial_data = model.spatial_location_series_data
        plot_epidemic_curves(series, save_path="epidemic_curves_mesa.png")
        plot_spatial_snapshot(spatial_data, save_path="spatial_animation_mesa.gif")
        plot_spatial_heatmap(
            spatial_data,
            save_path="spatial_heatmap_mesa.gif",
            world_size=model.world_size,
        )
        plot_spatial_3d_wave(
            spatial_data,
            save_path="infected_density_wave_mesa.gif",
            world_size=model.world_size,
        )


if __name__ == "__main__":
    main()

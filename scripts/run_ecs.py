from epidemic_sim.analysis.plotting import (
    plot_epidemic_curves,
    plot_spatial_3d_wave,
    plot_spatial_heatmap,
    plot_spatial_snapshot,
)
from epidemic_sim.ecs.simulation import SIREpidemicModel


def main():
    model = SIREpidemicModel(seed=28022026, enable_quarantine=False,
                             initial_infected=20, world_size=500,
                             beta_spatial=0.3, beta_network=0.25,
                             spatial_new=True, network_new=True,
                             space_attribute_similarity=True, n_agents=1000,
                             dt=1.0, dispersion=0.65)
    model.run(max_steps=400)
    time_series = model._collect_data()
    spatial_location_series = model.get_spatial_data()

    plot_epidemic_curves(time_series, save_path="epidemic_curves.png")
    plot_spatial_snapshot(spatial_location_series, save_path="spatial_animation.gif")
    plot_spatial_heatmap(model.spatial_location_series_data, save_path="spatial_heatmap.gif")
    plot_spatial_3d_wave(model.spatial_location_series_data,
                         save_path="wave_composite.gif", world_size=500,
                         compartment=None)


if __name__ == "__main__":
    main()

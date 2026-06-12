import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
import plotly.graph_objects as go

# --- GENERATE 3D PLOT ---
def generate_3d_raster_plot(firings, Ne, Ni, T, output_filename="3d_raster_plot.png",
                            downsample=None, show=False):
    """
    Generates and saves a 3D spatiotemporal raster plot.

    Args:
        firings (np.array): Array of shape [n_spikes, 2] with columns [time, neuron_index].
        Ne (int): Number of excitatory neurons.
        Ni (int): Number of inhibitory neurons.
        T (int): Total simulation time in ms.
        output_filename (str): The name of the file to save the plot.
        downsample (int): If set, randomly keep at most this many spikes for plotting.
        show (bool): If True, display the figure interactively (blocks); otherwise close it.
    """
    print("Generating 3D raster plot...")

    if downsample and len(firings) > downsample:
        idx = np.random.choice(len(firings), downsample, replace=False)
        firings = firings[idx]
        print(f"  Downsampled to {len(firings)} spikes")

    # --- 1. Assign spatial (X, Y) coordinates to each neuron ---
    total_neurons = Ne + Ni
    # Arrange all neurons on a square grid for simplicity.
    grid_side = int(np.ceil(np.sqrt(total_neurons)))

    neuron_coords = np.zeros((total_neurons, 2))
    for i in range(total_neurons):
        neuron_coords[i, 0] = i % grid_side  # x-coordinate
        neuron_coords[i, 1] = i // grid_side # y-coordinate

    # --- 2. Map spikes to the 3D space (T, X, Y) ---
    spike_times = firings[:, 0]
    fired_indices = firings[:, 1].astype(int)

    # Get the (x, y) position for each neuron that fired
    spike_x_pos = neuron_coords[fired_indices, 0]
    spike_y_pos = neuron_coords[fired_indices, 1]

    # --- 3. Create and save the 3D plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: x=time, y=neuron_x, z=neuron_y
    # Color points by time to show progression
    scatter = ax.scatter(spike_times, spike_x_pos, spike_y_pos, s=5, c=spike_times, cmap='plasma', marker='.')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron X-Position', fontsize=12)
    ax.set_zlabel('Neuron Y-Position', fontsize=12)
    ax.set_title('3D Spatiotemporal Raster Plot', fontsize=16)
    ax.set_xlim([0, T])
    ax.set_ylim([0, grid_side])
    ax.set_zlim([0, grid_side])

    # Add a color bar to indicate time
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Time (ms)')
    
    # Adjust viewing angle for a better perspective
    ax.view_init(elev=25, azim=-135)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"✅ 3D plot saved successfully to: {output_filename}")

import plotly.graph_objects as go

# --- GENERATE INTERACTIVE 3D PLOT ---
def generate_3d_raster_plotly(firings, Ne, Ni, T, output_filename="3d_raster_plot.html", show_browser=False):
    """
    Generates and saves a GPU-accelerated, interactive 3D raster plot using Plotly.

    Args:
        firings (np.array): Array of shape [n_spikes, 2] with columns [time, neuron_index].
        Ne (int): Number of excitatory neurons.
        Ni (int): Number of inhibitory neurons.
        T (int): Total simulation time in ms.
        output_filename (str): The name of the HTML file to save the plot.
    """
    print("Generating interactive 3D raster plot with Plotly (GPU-accelerated)...")

    # --- 1. Assign spatial (X, Y) coordinates (same as before) ---
    total_neurons = Ne + Ni
    grid_side = int(np.ceil(np.sqrt(total_neurons)))
    neuron_coords = np.zeros((total_neurons, 2))
    for i in range(total_neurons):
        neuron_coords[i, 0] = i % grid_side
        neuron_coords[i, 1] = i // grid_side

    # --- 2. Map spikes to 3D space (same as before) ---
    spike_times = firings[:, 0]
    fired_indices = firings[:, 1].astype(int)
    spike_x_pos = neuron_coords[fired_indices, 0]
    spike_y_pos = neuron_coords[fired_indices, 1]

    # --- 3. Create the interactive 3D plot using Plotly ---
    fig = go.Figure(data=[go.Scatter3d(
        x=spike_times,
        y=spike_x_pos,
        z=spike_y_pos,
        mode='markers',
        marker=dict(
            size=2,
            color=spike_times,         # Set color to be based on time
            colorscale='Viridis',      # Use the 'Viridis' colormap
            colorbar=dict(title='Time (ms)'),
            opacity=0.8
        )
    )])

    # --- 4. Update the layout (labels, title, etc.) ---
    fig.update_layout(
        title='Interactive 3D Spatiotemporal Raster Plot',
        scene=dict(
            xaxis_title='Time (ms)',
            yaxis_title='Neuron X-Position',
            zaxis_title='Neuron Y-Position'
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

    # --- 5. Save to an HTML file and/or show in browser ---
    fig.write_html(output_filename)
    # fig.show() # Uncomment this to automatically open the plot in your browser
        # ✅ Conditionally show the plot in the browser
    
    if show_browser:
        fig.show()

    print(f"✅ Interactive 3D plot saved successfully to: {output_filename}")


# if __name__ == '__main__':
#     # This block runs only when the script is executed directly
#     # It generates mock data to demonstrate the plotting function.

#     # --- Simulation Parameters for Mock Data ---
#     Ne_mock = 800
#     Ni_mock = 200
#     T_mock = 1000
#     total_neurons_mock = Ne_mock + Ni_mock

#     # --- Generate Mock 'firings' Data (simulating a traveling wave) ---
#     print("Generating mock data for a traveling wave pattern...")
#     mock_firings_list = []
#     for t in range(T_mock):
#         # Create a wave of activity that moves diagonally
#         center_neuron = int((t / T_mock) * total_neurons_mock)
#         # Neurons near the center are more likely to fire
#         for offset in range(-20, 20):
#             neuron_idx = (center_neuron + offset) % total_neurons_mock
#             if np.random.rand() < 0.5: # 50% chance to fire
#                 mock_firings_list.append([t, neuron_idx])
    
#     mock_firings = np.array(mock_firings_list)

#     # --- Generate and save the plot ---
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = f"raster_3d_mock_wave_{timestamp}.png"
    
#     generate_3d_raster_plot(mock_firings, Ne_mock, Ni_mock, T_mock, output_filename=filename)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from os.path import join


def animate_swarm_3d(data: pd.DataFrame, title_str: str = "Swarm Animation", save_path: str = None):
    """
    Animates 3D swarm movement over time from a DataFrame with columns: time, x, y, z.

    Args:
        data (pd.DataFrame): Data with columns ['time', 'x', 'y', 'z']
        title_str (str): Title of the plot
        save_path (str): If provided, saves the animation to this path (GIF or MP4)
    """
    frames = int(data['time'].max()) + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    graph, = ax.plot([], [], [], 'b.', alpha=0.8)
    title = ax.set_title(title_str)

    ax.set_xlim(data.x.min(), data.x.max())
    ax.set_ylim(data.y.min(), data.y.max())
    ax.set_zlim(data.z.min(), data.z.max())

    def update_graph(frame):
        frame_data = data[data['time'] == frame]
        graph.set_data(frame_data.x, frame_data.y)
        graph.set_3d_properties(frame_data.z)
        title.set_text(f"{title_str}, time={frame}")
        return title, graph

    anim = animation.FuncAnimation(fig, update_graph, frames=frames, interval=300, blit=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer='imagemagick')
        else:
            anim.save(save_path, writer='ffmpeg')

    plt.show()


if __name__ == "__main__":
    # Simulated example data
    a = np.random.rand(2000, 3) * 10
    t = np.repeat(np.arange(20), 100)
    df = pd.DataFrame({'time': t, 'x': a[:, 0], 'y': a[:, 1], 'z': a[:, 2]})

    animate_swarm_3d(df, title_str="Simulated Swarm Movement", save_path=None)  # Change to 'swarm.gif' to save

#To use output call
#df = pd.read_csv("your_swarm_output.csv")
#animate_swarm_3d(df, title_str="COMB-PSO Feature Space", save_path="swarm.mp4")

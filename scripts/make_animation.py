import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    file = "simple_ba.npy"
    history = np.load(file)
    
    # Initialize Animation
    points = history[:, 48:].reshape(-1, 15, 3)
    scat = ax.scatter(points[0][:, 0], points[0][:, 1], points[0][:, 2], color='blue')
            
    # Update function for animation
    def update(frame):
        scat._offsets3d = (points[frame][:, 0], points[frame][:, 1], points[frame][:, 2])
        return scat,
    
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=1, blit=False)

    # Set the labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Optimization Process')
    
    # Show the animation
    plt.show()

    # ani.save('optimization_process.gif', writer='pillow', fps=60)

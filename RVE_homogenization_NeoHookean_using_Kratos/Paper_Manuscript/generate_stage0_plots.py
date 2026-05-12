import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from plot_style_utils import apply_latex_plot_style
apply_latex_plot_style()

def scaled_path(path_e):
    p = np.array(path_e, dtype=float)
    p[:, 2] *= 0.5
    return p

def create_base_plot(title, draw_ellipsoid=False):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("Exx [-]")
    ax.set_ylabel("Eyy [-]")
    ax.set_zlabel("Gxy/2 [-]")
    
    # Establish consistent limits and TRUE aspect ratio for the new 50% tension domain
    mins = np.array([-0.15, -0.15, -0.30])
    maxs = np.array([ 0.65,  0.65,  0.30])
    
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    
    # Crucial step: forces visual aspect ratio to rigidly match data aspect ratio
    ax.set_box_aspect((maxs[0]-mins[0], maxs[1]-mins[1], maxs[2]-mins[2]))
    
    # Draw Origin
    ax.scatter([0], [0], [0], color='black', s=50, marker='X', label='Start (0,0,0)', zorder=10)

    if draw_ellipsoid:
        # True bounded limits from asymmetric compression/tension boundary constraints
        # relative = [1.  0.2 1.  0.2 0.5 0.5 ]
        # emax = 0.5
        exx_pos, exx_neg = 0.5, 0.1
        eyy_pos, eyy_neg = 0.5, 0.1
        gxy_pos, gxy_neg = 0.25, 0.25
        
        c = np.array([0.5*(exx_pos-exx_neg), 0.5*(eyy_pos-eyy_neg), 0.5*(gxy_pos-gxy_neg)])
        a = np.array([0.5*(exx_pos+exx_neg), 0.5*(eyy_pos+eyy_neg), 0.5*(gxy_pos+gxy_neg)])
        
        # Scaling the physical Z axis explicitly
        c[2] *= 0.5
        a[2] *= 0.5
        
        u = np.linspace(0.0, 2.0 * np.pi, 56)
        v = np.linspace(0.0, np.pi, 28)
        uu, vv = np.meshgrid(u, v)
        x = c[0] + a[0] * np.cos(uu) * np.sin(vv)
        y = c[1] + a[1] * np.sin(uu) * np.sin(vv)
        z = c[2] + a[2] * np.cos(vv)
        ax.plot_surface(x, y, z, color="gray", alpha=0.20, linewidth=0.0, zorder=-1)
        
    ax.view_init(elev=24.0, azim=38.0)
        
    return fig, ax

def plot_stage0_auxplots(npz_file):
    data = np.load(npz_file)
    traj_count = int(data['trajectory_count'])
    labels = data['trajectory_labels']
    
    # PLOT 1: 2E12 = 0 plane (Gxy = 0) -> Trajectories 1 and 2
    fig1, ax1 = create_base_plot("Equatorial Shear Plane ($G_{xy} = 0$)")
    for idx in [1, 2]:
        traj = scaled_path(data[f'trajectory_{idx}'])
        col = 'tab:blue' if idx == 1 else 'tab:orange'
        ax1.plot(traj[:,0], traj[:,1], traj[:,2], color=col, alpha=0.8, linewidth=2.0)
        ax1.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', s=40)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("stage0_trajectories_plane_0.png", dpi=300)

    # PLOT 2: Positive 2E12 plane (Gxy = +0.25) -> Trajectories 7 and 8
    fig2, ax2 = create_base_plot("Positive Shear Plane ($G_{xy} = +0.25$)")
    for idx in [7, 8]:
        traj = scaled_path(data[f'trajectory_{idx}'])
        col = 'tab:blue' if idx == 7 else 'tab:orange'
        ax2.plot(traj[:,0], traj[:,1], traj[:,2], color=col, alpha=0.8, linewidth=2.0)
        ax2.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', s=40)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("stage0_trajectories_plane_pos.png", dpi=300)

    # PLOT 3: Negative 2E12 plane (Gxy = -0.25) -> Trajectories 17 and 18
    fig3, ax3 = create_base_plot("Negative Shear Plane ($G_{xy} = -0.25$)")
    for idx in [17, 18]:
        traj = scaled_path(data[f'trajectory_{idx}'])
        col = 'tab:blue' if idx == 17 else 'tab:orange'
        ax3.plot(traj[:,0], traj[:,1], traj[:,2], color=col, alpha=0.8, linewidth=2.0)
        ax3.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', s=40)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig("stage0_trajectories_plane_neg.png", dpi=300)

    # PLOT 4: ALL trajectories combined + Opaque Ellipsoid
    fig4, ax4 = create_base_plot("Stage 0 Final Overview", draw_ellipsoid=True)
    for i in range(1, traj_count + 1):
        traj = scaled_path(data[f'trajectory_{i}'])
        col = 'tab:blue' if (i-1)%2 == 0 else 'tab:orange'
        ax4.plot(traj[:,0], traj[:,1], traj[:,2], color=col, alpha=1.0, linewidth=1.3)
        ax4.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', s=10)
        
    ax4.legend(loc="upper left")
    fig4.tight_layout()
    fig4.savefig("stage0_trajectories_all_ellipsoid.png", dpi=300)
    
    print("Saved all 4 correctly scaled Phase 0 visualizations.")

if __name__ == "__main__":
    npz_file = "../stage_0_trajectory/stage_0_trajectories.npz"
    plot_stage0_auxplots(npz_file)

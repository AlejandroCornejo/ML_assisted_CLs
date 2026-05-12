import os
import numpy as np
import pyvista as pv

def main():
    mdpa_file = "../rve_geometry.mdpa"
    
    nodes = {}
    elements = []

    in_nodes = False
    in_elements = False

    with open(mdpa_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("Begin Nodes"):
                in_nodes = True
                continue
            elif line.startswith("End Nodes"):
                in_nodes = False
                continue
            
            if line.startswith("Begin Geometries Triangle2D6"):
                in_elements = True
                continue
            elif line.startswith("End Geometries"):
                in_elements = False
                continue
            
            if in_nodes:
                parts = line.split()
                node_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                nodes[node_id] = (x, y)
                
            if in_elements:
                parts = line.split()
                n1, n2, n3, n4, n5, n6 = [int(p) for p in parts[1:7]]
                # PyVista 6-node quadratic triangle wants node order 0, 1, 2, 3, 4, 5
                elements.append([n1, n2, n3, n4, n5, n6])

    # Convert nodes to array
    max_node_id = max(nodes.keys())
    coords = np.zeros((max_node_id + 1, 3), dtype=np.float64)
    for nid, (x, y) in nodes.items():
        coords[nid, 0] = x
        coords[nid, 1] = y
        
    node_id_to_idx = {nid: nid for nid in nodes.keys()} # Direct mapping for simplicity

    # Prepare PyVista data
    cell_blocks = []
    cell_types = []
    
    for elem in elements:
        npts = 6
        conn = elem
        ctype = pv.CellType.QUADRATIC_TRIANGLE
        cell_blocks.append(np.array([npts] + conn, dtype=np.int64))
        cell_types.append(ctype)

    cells = np.concatenate(cell_blocks)
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    mesh = pv.UnstructuredGrid(cells, cell_types, coords)
    
    # Plotting
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    plotter.set_background("white")
    
    plotter.add_mesh(
        mesh,
        color="#f0f0f0",
        show_edges=True,
        edge_color="black",
        line_width=1.0
    )
    
    plotter.view_xy()
    plotter.enable_parallel_projection()
    
    title_txt = "RVE Domain and Mesh\n(Quadratic Triangles)"
    plotter.add_text(title_txt, position="upper_edge", font_size=18, color="black")
    
    out_path = "mesh_plot.png"
    plotter.screenshot(out_path)
    print(f"PyVista mesh plot saved to {out_path}")

if __name__ == "__main__":
    main()

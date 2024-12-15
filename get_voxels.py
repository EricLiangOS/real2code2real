import numpy as np
import trimesh
import torch
import open3d as o3d

def obj_to_voxel_tensor(obj_path, grid_size=64):
    """
    Convert an .obj file to a binary voxel tensor of specified grid size.
    
    Parameters:
    -----------
    obj_path : str
        Path to the .obj file
    grid_size : int, optional
        Size of the 3D voxel grid (default is 64)
    
    Returns:
    --------
    torch.Tensor
        Binary tensor representing the voxelized 3D object
    """
    # Load the mesh
    mesh = trimesh.load(obj_path)
    
    # Compute the bounding box of the mesh
    bounds = mesh.bounding_box.bounds
    
    # Calculate the scale and translation to fit the object in the grid
    mesh_min = bounds[0]
    mesh_max = bounds[1]
    mesh_size = mesh_max - mesh_min
    
    # Determine the maximum dimension to ensure proper scaling
    max_dimension = np.max(mesh_size)
    
    # Scale factor to fit the object in the grid
    scale_factor = (grid_size - 1) / max_dimension
    
    # Create a transformation matrix
    transform = np.eye(4)
    transform[:3, :3] *= scale_factor
    transform[:3, 3] = -mesh_min * scale_factor
    
    # Apply the transformation to center and scale the mesh
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(transform)
    
    # Create a voxel grid
    voxel_grid = transformed_mesh.voxelized(pitch=1.0).matrix
    
    # Pad or crop the voxel grid to exactly match the grid size
    if voxel_grid.shape[0] > grid_size or voxel_grid.shape[1] > grid_size or voxel_grid.shape[2] > grid_size:
        voxel_grid = voxel_grid[:grid_size, :grid_size, :grid_size]
    else:
        pad_needed = grid_size - np.array(voxel_grid.shape)
        voxel_grid = np.pad(voxel_grid, 
                             ((0, pad_needed[0]), 
                              (0, pad_needed[1]), 
                              (0, pad_needed[2])), 
                             mode='constant', 
                             constant_values=0)
    
    # Convert to binary tensor (bool or float)
    voxel_tensor = torch.tensor(voxel_grid > 0, dtype=torch.float32)
    
    return voxel_tensor

def save_np_to_voxels(voxels):
    grid = voxels.detach().cpu().numpy()

    # Create new voxel grid object and set voxel_size to some value
    # --> otherwise it will default to 0 and the grid will be invisible
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
    # Iterate over numpy grid
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                if grid[x, y, z] == 0:
                    continue
                # Create a voxel object
                voxel = o3d.geometry.Voxel()
                voxel.color = np.array([0.0, 0.0, 0.0]) * (1.0 - grid[x, y, z])
                voxel.grid_index = np.array([x, y, z])
                # Add voxel object to grid
                voxel_grid.add_voxel(voxel)
    o3d.io.write_voxel_grid("voxels.ply", voxel_grid)

# Example usage
def main():
    # Replace 'path/to/your/model.obj' with the actual path to your .obj file
    obj_path = '/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/blender_meshes/link_1.obj'
    
    # Convert OBJ to voxel tensor
    voxel_tensor = obj_to_voxel_tensor(obj_path)
    
    # Print some information about the tensor
    print("Voxel Tensor Shape:", voxel_tensor.shape)
    print("Total Voxels:", voxel_tensor.sum().item())
    print("Voxel Tensor Type:", voxel_tensor.dtype)
    print("done")
    save_np_to_voxels(voxel_tensor)

if __name__ == '__main__':
    main()
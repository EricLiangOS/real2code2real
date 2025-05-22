
import open3d as o3d
import numpy as np

def align_mesh_to_pcd_aabb_extents(
    mesh: o3d.geometry.TriangleMesh,
    point_cloud: o3d.geometry.PointCloud,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    min_cleaned_pcd_points: int = 100 
) -> o3d.geometry.TriangleMesh:
    """
    Aligns and scales a mesh to match the axis-aligned bounding box (AABB)
    center and extents of a given point cloud. The point cloud is first
    cleaned using statistical outlier removal to make the AABB robust to outliers.

    The process involves:
    1. Cleaning the input point cloud using statistical outlier removal.
    2. Calculating the AABB for the mesh and the cleaned point cloud.
    3. Determining scale factors for each axis (X, Y, Z) based on the
       ratio of cleaned point cloud AABB extents to mesh AABB extents.
    4. Translating the mesh so its AABB center is at the origin.
    5. Scaling the mesh vertices by the calculated scale factors.
    6. Translating the scaled mesh so its new AABB center aligns with
       the cleaned point cloud's AABB center.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to be aligned and scaled.
        point_cloud (o3d.geometry.PointCloud): The reference point cloud.
        sor_nb_neighbors (int): Number of neighbors to use for statistical outlier removal.
        sor_std_ratio (float): Standard deviation ratio for statistical outlier removal.
        min_cleaned_pcd_points (int): Minimum number of points required in the
                                      cleaned point cloud to use its AABB.
                                      Otherwise, the original point cloud's AABB is used.


    Returns:
        o3d.geometry.TriangleMesh: A new mesh, aligned and scaled.
    
    Raises:
        TypeError: If inputs are not of the correct Open3D types.
        ValueError: If mesh is empty, or if mesh AABB has zero extent.
                     If the original point cloud is empty.
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("Input 'mesh' must be an Open3D TriangleMesh.")
    
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise TypeError("Input 'point_cloud' must be an Open3D PointCloud.")
    if mesh.is_empty():
        raise ValueError("Input 'mesh' is empty.")
    if not point_cloud.has_points():
        raise ValueError("Input 'point_cloud' is empty.")

    # Create a copy of the mesh to avoid modifying the original
    aligned_mesh = o3d.geometry.TriangleMesh(mesh)
    
   # 1. Clean the point cloud and get its AABB
    pcd_to_use = point_cloud
    if point_cloud.has_points():
        print(f"Original point cloud has {len(point_cloud.points)} points.")
        # Create a copy for cleaning
        pcd_copy_for_cleaning = o3d.geometry.PointCloud(point_cloud)
        cleaned_pcd, ind = pcd_copy_for_cleaning.remove_statistical_outlier(
            nb_neighbors=sor_nb_neighbors, 
            std_ratio=sor_std_ratio
        )
        print(f"Point cloud after statistical outlier removal has {len(cleaned_pcd.points)} points.")
        if len(cleaned_pcd.points) >= min_cleaned_pcd_points:
            pcd_to_use = cleaned_pcd
            print("Using cleaned point cloud for AABB calculation.")
        else:
            print(f"Warning: Cleaned point cloud has only {len(cleaned_pcd.points)} points (less than min_cleaned_pcd_points={min_cleaned_pcd_points}). Using original point cloud for AABB.")
            # pcd_to_use remains the original point_cloud
    else: # Should have been caught by initial check, but as a safeguard
        raise ValueError("Input 'point_cloud' is empty, cannot proceed.")


    mesh_aabb = aligned_mesh.get_axis_aligned_bounding_box()
    pcd_aabb = pcd_to_use.get_axis_aligned_bounding_box()

    mesh_center = mesh_aabb.get_center()
    mesh_extent = mesh_aabb.get_extent()

    pcd_center = pcd_aabb.get_center()
    pcd_extent = pcd_aabb.get_extent()

    if np.any(np.isclose(mesh_extent, 0)):
        raise ValueError("Mesh AABB has zero extent along one or more axes. Cannot determine scale factors.")
    if np.any(np.isclose(pcd_extent, 0)):
        # This might happen if cleaning is too aggressive or PCD is degenerate
        print("Warning: Point cloud AABB (after potential cleaning) has zero extent along one or more axes.")
        if pcd_to_use is not point_cloud and point_cloud.has_points(): # if cleaning was attempted and failed, try original
            print("Falling back to original point cloud AABB for extents.")
            pcd_aabb_original = point_cloud.get_axis_aligned_bounding_box()
            pcd_center = pcd_aabb_original.get_center() # Keep center from cleaned if possible, or use original
            pcd_extent = pcd_aabb_original.get_extent()
            if np.any(np.isclose(pcd_extent, 0)):
                 raise ValueError("Original point cloud AABB also has zero extent. Cannot determine scale factors.")
        else:
            raise ValueError("Point cloud AABB has zero extent. Cannot determine scale factors.")



    # 1. Get AABBs
    mesh_aabb = aligned_mesh.get_axis_aligned_bounding_box()
    pcd_aabb = point_cloud.get_axis_aligned_bounding_box()

    mesh_center = mesh_aabb.get_center()
    mesh_extent = mesh_aabb.get_extent()

    pcd_center = pcd_aabb.get_center()
    pcd_extent = pcd_aabb.get_extent()

    if np.any(np.isclose(mesh_extent, 0)):
        raise ValueError("Mesh AABB has zero extent along one or more axes. Cannot determine scale factors.")

    # 2. Calculate scale factors
    scale_factors = pcd_extent / mesh_extent
    
    # 3. Transform vertices:
    vertices = np.asarray(aligned_mesh.vertices)
    
    #   a. Translate mesh vertices so its AABB center is at the origin.
    vertices_centered = vertices - mesh_center
    
    #   b. Scale the centered vertices.
    vertices_scaled = vertices_centered * scale_factors
    
    #   c. Translate the scaled vertices to the point cloud's AABB center.
    vertices_final = vertices_scaled + pcd_center
    
    # Update mesh vertices
    aligned_mesh.vertices = o3d.utility.Vector3dVector(vertices_final)
    
    # Recompute normals as the mesh shape has changed
    if aligned_mesh.has_vertex_normals() or aligned_mesh.has_triangle_normals():
        aligned_mesh.compute_vertex_normals()
        aligned_mesh.compute_triangle_normals(normalized=True)

    return aligned_mesh

# Example Usage:
if __name__ == '__main__':
    # --- Load Source Mesh from file ---
    # IMPORTANT: Replace with your actual mesh file path
    source_mesh_path = "/store/real/ehliang/multiview_outputs/kitchen_synthesizer_2/20_img_30_steps_7.5_sparse_3_latent/object_3/state_1/object_3_state_1_aligned_mesh.obj"
    try:
        source_mesh = o3d.io.read_triangle_mesh(source_mesh_path)
        if source_mesh.is_empty():
            print(f"Error: Could not read source mesh from {source_mesh_path} or it is empty.")
            exit()
        if not source_mesh.has_vertex_normals(): # Ensure normals for shading
            source_mesh.compute_vertex_normals()
        source_mesh.paint_uniform_color([0.8, 0.6, 0.4]) # Orange-ish
    except Exception as e:
        print(f"Error loading source mesh: {e}")
        exit()

    # --- Load Target Point Cloud from file ---
    # IMPORTANT: Replace with your actual point cloud file path
    target_pcd_path = "/store/real/ehliang/multiview_outputs/kitchen_synthesizer_2/20_img_30_steps_7.5_sparse_3_latent/object_3/state_1/object_3_state_1_ground_truth_pcd.ply"
    try:
        target_pcd = o3d.io.read_point_cloud(target_pcd_path)
        if not target_pcd.has_points():
            print(f"Error: Could not read target point cloud from {target_pcd_path} or it is empty.")
            exit()
        target_pcd.paint_uniform_color([0.4, 0.6, 0.8]) # Blue-ish
    except Exception as e:
        print(f"Error loading target point cloud: {e}")
        exit()

    print("--- Original Source Mesh ---")
    source_mesh_aabb = source_mesh.get_axis_aligned_bounding_box()
    source_mesh_aabb.color = (1,0,0) # Red AABB
    print(f"  AABB Center: {source_mesh_aabb.get_center()}")
    print(f"  AABB Extent: {source_mesh_aabb.get_extent()}")

    print("\n--- Target Point Cloud ---")
    target_pcd_aabb = target_pcd.get_axis_aligned_bounding_box()
    target_pcd_aabb.color = (0,1,0) # Green AABB
    print(f"  AABB Center: {target_pcd_aabb.get_center()}")
    print(f"  AABB Extent: {target_pcd_aabb.get_extent()}")

    try:
        # Perform the alignment and scaling
        aligned_scaled_mesh = align_mesh_to_pcd_aabb_extents(source_mesh, target_pcd)
        aligned_scaled_mesh.paint_uniform_color([0.6, 0.8, 0.4]) # Green-ish for the new mesh

        print("\n--- Aligned and Scaled Mesh (AABB method) ---")
        aligned_mesh_aabb = aligned_scaled_mesh.get_axis_aligned_bounding_box()
        aligned_mesh_aabb.color = (0,0,1) # Blue AABB
        print(f"  AABB Center: {aligned_mesh_aabb.get_center()}")
        print(f"  AABB Extent: {aligned_mesh_aabb.get_extent()}")

        # Verification
        print("\n--- Verification ---")
        print(f"  Aligned mesh AABB center close to target PCD AABB center? {np.allclose(aligned_mesh_aabb.get_center(), target_pcd_aabb.get_center())}")
        print(f"  Aligned mesh AABB extent close to target PCD AABB extent? {np.allclose(aligned_mesh_aabb.get_extent(), target_pcd_aabb.get_extent())}")
        
        # --- Save the Aligned and Scaled Mesh ---
        # IMPORTANT: Choose an appropriate output path and name
        output_aligned_mesh_path = "/store/real/ehliang/multiview_outputs/kitchen_synthesizer_2/20_img_30_steps_7.5_sparse_3_latent/object_3/state_1/object_3_state_1_aabb_aligned_scaled_mesh.obj"
        try:
            o3d.io.write_triangle_mesh(output_aligned_mesh_path, aligned_scaled_mesh, write_vertex_normals=True)
            print(f"\nSuccessfully saved aligned and scaled mesh to: {output_aligned_mesh_path}")
        except Exception as e_save:
            print(f"\nError saving aligned and scaled mesh: {e_save}")

        # --- Visualization ---
        source_mesh_viz = o3d.geometry.TriangleMesh(source_mesh)
        target_pcd_viz = o3d.geometry.PointCloud(target_pcd)
        aligned_scaled_mesh_viz = o3d.geometry.TriangleMesh(aligned_scaled_mesh)
        
        # For clearer visualization, shift geometries apart
        x_shift = target_pcd_aabb.get_extent()[0] * 1.2 # Shift by a bit more than target width
        
        source_mesh_viz.translate([-x_shift, 0, 0], relative=True)
        source_mesh_aabb_viz = source_mesh_viz.get_axis_aligned_bounding_box()
        source_mesh_aabb_viz.color = (1,0,0) # Red

        target_pcd_viz.translate([x_shift, 0, 0], relative=True)
        target_pcd_aabb_viz = target_pcd_viz.get_axis_aligned_bounding_box()
        target_pcd_aabb_viz.color = (0,1,0) # Green
        
        # Aligned mesh is already where we want it for visualization (center)
        aligned_mesh_aabb_viz = aligned_scaled_mesh_viz.get_axis_aligned_bounding_box()
        aligned_mesh_aabb_viz.color = (0,0,1) # Blue

        o3d.visualization.draw_geometries([
            source_mesh_viz, source_mesh_aabb_viz,
            aligned_scaled_mesh_viz, aligned_mesh_aabb_viz,
            target_pcd_viz, target_pcd_aabb_viz
        ], window_name="Original Mesh (left), AABB Aligned/Scaled Mesh (center), Target PCD (right)")

    except Exception as e:
        print(f"An error occurred during alignment/scaling: {e}")
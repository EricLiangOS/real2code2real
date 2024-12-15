import open3d as o3d
import numpy as np

def remove_smaller_clusters(mesh):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)

    return mesh

def remove_black_spots(mesh, black_threshold = 0.025):
    mesh.compute_vertex_normals()

    vertex_colors = np.asarray(mesh.vertex_colors)
    vertices = np.asarray(mesh.vertices)

    is_black = np.all(vertex_colors < black_threshold, axis=1)

    triangles = np.asarray(mesh.triangles)
    triangle_mask = np.any(is_black[triangles], axis=1)

    mesh.remove_triangles_by_mask(triangle_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


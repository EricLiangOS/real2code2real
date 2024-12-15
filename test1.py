import trimesh
import numpy as np
import argparse
import os

def extract_mesh_colors(mesh):
    """
    Attempt to extract vertex colors from a mesh using multiple methods.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh to extract colors from
    
    Returns:
    --------
    numpy.ndarray or None
        Vertex colors if found, otherwise None
    """
    # Try different color extraction methods
    color_methods = [
        # Try direct vertex colors
        lambda m: m.visual.vertex_colors if hasattr(m, 'visual') and m.visual.kind == 'vertex' else None,
        
        # Try face colors (convert to vertex colors)
        lambda m: m.visual.face_colors if hasattr(m, 'visual') and m.visual.kind == 'face' else None,
        
        # Try to get colors from mesh attributes
        lambda m: m.metadata.get('vertex_colors') if hasattr(m, 'metadata') else None,
        
        # Try to get colors from visual.vertex
        lambda m: getattr(m.visual, 'vertex', None)
    ]
    
    # Try each method
    for method in color_methods:
        colors = method(mesh)
        
        # Validate colors
        if colors is not None and len(colors) > 0:
            # Ensure we have 4 channels (RGBA)
            if colors.shape[1] == 3:
                # If RGB, add full opacity
                colors = np.column_stack([colors, np.full((len(colors), 1), 255)])
            
            return colors
    
    return None

def convert_glb_to_ply(input_glb, output_ply=None):
    """
    Convert a GLB file to a PLY file, preserving vertex colors.
    
    Parameters:
    -----------
    input_glb : str
        Path to the input GLB file
    output_ply : str, optional
        Path to the output PLY file. If not provided, 
        will use the input filename with .ply extension
    
    Returns:
    --------
    str
        Path to the converted PLY file
    """
    # Load the scene or mesh
    scene_or_mesh = trimesh.load(input_glb, force='mesh')
    
    # Determine output filename
    if output_ply is None:
        output_ply = os.path.splitext(input_glb)[0] + '.ply'
    
    # Handle different types of loaded objects
    if isinstance(scene_or_mesh, trimesh.Scene):
        # If it's a scene, combine all meshes
        meshes = list(scene_or_mesh.geometry.values())
        
        if not meshes:
            raise ValueError("No meshes found in the scene")
        
        # Combine meshes if there are multiple
        if len(meshes) > 1:
            # Attempt to preserve colors during combination
            combined_vertices = []
            combined_faces = []
            all_vertex_colors = []
            
            vertex_offset = 0
            for mesh in meshes:
                # Extract vertices and faces
                mesh_vertices = mesh.vertices
                mesh_faces = mesh.faces.copy()
                
                # Adjust face indices
                mesh_faces += vertex_offset
                
                # Extract colors
                mesh_colors = extract_mesh_colors(mesh)
                
                # Append to combined lists
                combined_vertices.append(mesh_vertices)
                combined_faces.append(mesh_faces)
                
                if mesh_colors is not None:
                    all_vertex_colors.append(mesh_colors)
                
                # Update vertex offset
                vertex_offset += len(mesh_vertices)
            
            # Combine vertices and faces
            combined_mesh = trimesh.Trimesh(
                vertices=np.concatenate(combined_vertices),
                faces=np.concatenate(combined_faces)
            )
            
            # Combine vertex colors if they exist
            if all_vertex_colors:
                combined_colors = np.concatenate(all_vertex_colors)
                combined_mesh.visual.vertex_colors = combined_colors
            
            mesh = combined_mesh
        else:
            # Single mesh in the scene
            mesh = meshes[0]
    else:
        # If it's already a single mesh
        mesh = scene_or_mesh
    
    # Extract vertex colors
    vertex_colors = extract_mesh_colors(mesh)
    
    if vertex_colors is not None:
        # Export mesh with vertex colors to PLY
        mesh.export(output_ply, file_type='ply', 
                    vertex_attributes={
                        'red': vertex_colors[:, 0],
                        'green': vertex_colors[:, 1],
                        'blue': vertex_colors[:, 2],
                        'alpha': vertex_colors[:, 3]
                    })
        
        print(f"Successfully converted {input_glb} to {output_ply}")
        print(f"Vertex count: {len(mesh.vertices)}")
        print(f"Vertex color information preserved")
        
    else:
        # If no vertex colors, export as-is
        mesh.export(output_ply)
        print(f"Converted {input_glb} to {output_ply}")
        print(f"Note: No vertex color information found in the original mesh")
    
    return output_ply

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert GLB file to PLY with vertex colors')
    parser.add_argument('--input', help='Path to input GLB file')
    parser.add_argument('-o', '--output', help='Path to output PLY file (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert the file
    convert_glb_to_ply(args.input, args.output)

if __name__ == '__main__':
    main()

# Example usage:
# python glb_to_ply_converter.py input.glb
# python glb_to_ply_converter.py input.glb -o output.ply
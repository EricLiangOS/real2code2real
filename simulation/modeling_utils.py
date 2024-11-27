import os
import sys
import bpy
import argparse
from PIL import Image
import trimesh

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_xml(texture_path, object_path, output_path):
    # Step 5: Create MuJoCo XML

    object_name = os.path.basename(output_path).split(".")[0]

    xml_content = f"""
    <mujoco>
        <asset>
            <mesh name="{object_name}_mesh" file="{object_path}"/>
            <texture name="{object_name}_texture" type="2d" file="{texture_path}"/>
        </asset>
        <worldbody>
            <body name="object">
                <geom type="mesh" mesh="{object_name}_mesh" material="object_material"/>
            </body>
        </worldbody>
        <asset>
            <material name="object_material" texture="{object_name}_texture" specular="0.5" shininess="0.5"/>
        </asset>
    </mujoco>
    """
    with open(output_path, 'w') as f:
        f.write(xml_content)

# convert ply file with vertex colors into an obj file with vertex colors
def convert_ply_to_obj(ply_path, output_path):
    mesh = trimesh.load(ply_path, file_type='ply')
    
    # Extract vertices, colors, and faces
    vertices = mesh.vertices
    colors = mesh.visual.vertex_colors[:, :3]  # Extract RGB, ignore alpha
    faces = mesh.faces

    # Write to OBJ file
    with open(output_path, 'w') as obj_file:
        # Write vertices with color
        for i, (vertex, color) in enumerate(zip(vertices, colors)):
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {color[0]/255:.6f} {color[1]/255:.6f} {color[2]/255:.6f}\n")

        # Write faces (OBJ indices are 1-based)
        for face in faces:
            obj_file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def extract_texture_from_obj(input_path, output_path):
    def clear_scene():
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
    
    clear_scene()

    bpy.ops.wm.obj_import(filepath=input_path, directory=os.path.split(input_path)[0], files=[{"name": os.path.split(input_path)[1]}])
    bpy.context.object.rotation_euler[0] = 0
    obj = bpy.context.active_object

    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.editmode_toggle()

    # vertex color material
    mat = bpy.data.materials.new(name="VertexColor")
    mat.use_nodes = True
    vc = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    vc.name = "vc_node"
    vc.layer_name = "Color"
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    mat.node_tree.links.new(vc.outputs[0], bsdf.inputs[0])
    obj.data.materials.append(mat)

    # bake
    image_name = obj.name + '_BakedTexture'
    img = bpy.data.images.new(image_name, 1024, 1024)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_selected_to_active = False

    image_name = obj.name + '_BakedTexture'
    img = bpy.data.images.new(image_name, 1024, 1024)

    for mat in obj.data.materials:

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        texture_node =nodes.new('ShaderNodeTexImage')
        texture_node.name = 'Bake_node'
        texture_node.select = True
        nodes.active = texture_node
        texture_node.image = img
        
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='DIFFUSE', save_mode='EXTERNAL')

    # remove original color on vertex node.
    for mat in obj.data.materials:
        vc = mat.node_tree.nodes['vc_node']
        mat.node_tree.nodes.remove(vc)

    object_name = os.path.basename(input_path).split(".")[0]

    img.save_render(filepath=f'{os.path.join(output_path, "textures")}/{object_name}.png')

    if bpy.app.version[0] >= 4:
        bpy.ops.wm.obj_export(filepath=f"{os.path.join(output_path, 'meshes')}/{object_name}.obj")
    else:
        bpy.ops.export_scene.obj(filepath=f"{os.path.join(output_path, 'meshes')}/{object_name}.obj")

def generate_mujoco_files(input_dir):
    assets_path = os.path.join(input_dir, "assets")
    meshes_dir = os.path.join(assets_path, "meshes")
    textures_dir = os.path.join(assets_path, "textures")
    
    mesh_files = sorted([f for f in os.listdir(meshes_dir) if f.endswith(".obj")])
    texture_files = sorted([f for f in os.listdir(textures_dir) if f.endswith(".png")])
    
    # Generate object_assets.xml
    assets_xml = ["<mujocoinclude>", "  <asset>"]
    for i, (mesh_file, texture_file) in enumerate(zip(mesh_files, texture_files), start=1):
        object_name = f"object_{i}"
        mesh_path = f"meshes/{mesh_file}"
        texture_path = f"textures/{texture_file}"
        assets_xml.append(f'    <mesh name="{object_name}_mesh" file="{mesh_path}" />')
        assets_xml.append(f'    <texture name="{object_name}_texture" type="2d" file="{texture_path}" />')
        assets_xml.append(f'    <material name="{object_name}_material" reflectance="0.5" texture="{object_name}_texture" />\n')

    assets_xml.append('    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>')
    assets_xml.append('    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="2 2" reflectance="0.2"/>')

    assets_xml.append("  </asset>")
    assets_xml.append("</mujocoinclude>")

    with open(os.path.join(assets_path, "object_assets.xml"), "w") as f:
        f.write("\n".join(assets_xml))
    
    # Generate scene.xml
    scene_xml = [
        '  <mujoco model="object_scene"  >',
        '  <include file="assets/object_assets.xml" />\n',
        '  <visual>',
        '    <headlight diffuse="0.6 0.6 0.6"  ambient="0.1 0.1 0.1" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000"/>',
        '    <quality shadowsize="4096"/> ',
        '  </visual>\n',
        "  <worldbody>",
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>',
        '    <geom name="floor" pos="0 0 -5" type="plane" size="0 0 0.05" material="groundplane" />\n'
    ]
    
    for i in range(1, len(mesh_files) + 1):
        object_name = f"object_{i}"
        scene_xml.append(f'    <body name="{object_name}_body">')
        scene_xml.append(f'      <freejoint />')
        scene_xml.append(f'      <geom type="mesh" mesh="{object_name}_mesh" material="{object_name}_material" />\n')
        scene_xml.append(f'    </body>')
    
    scene_xml.append("  </worldbody>")
    scene_xml.append("</mujoco>")
    
    with open(os.path.join(input_dir, "scene.xml"), "w") as f:
        f.write("\n".join(scene_xml))

def get_objects_from_directory(input_dir):
    
    assets_path = os.path.join(input_dir, "envs", "assets")
    raw_object_path = os.path.join(assets_path, "objects")
    meshes_path = os.path.join(assets_path, "meshes")
    textures_path = os.path.join(assets_path, "textures")
    
    os.makedirs(assets_path, exist_ok=True)
    os.makedirs(raw_object_path, exist_ok=True)
    os.makedirs(meshes_path, exist_ok=True)
    os.makedirs(textures_path, exist_ok=True)

    obj_list = [ item for item in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, item)) ]

    for obj in obj_list:
        ply_file = os.path.join(input_dir, obj, "output", f"{obj}.ply")
        raw_object = os.path.join(raw_object_path, f"{obj}.obj")

        if not os.path.isfile(ply_file) or (obj != "object_3" and obj != "object_1"):
            continue
        
        convert_ply_to_obj(ply_file, raw_object)
        extract_texture_from_obj(raw_object, assets_path)

    generate_mujoco_files(os.path.join(input_dir, "envs"))

    print(bcolors.OKGREEN + f"Object XMLs successfully saved at {assets_path}" + bcolors.ENDC)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", "-o", required=True, type=str)
    args = parser.parse_args()

    get_objects_from_directory(args.object_path)


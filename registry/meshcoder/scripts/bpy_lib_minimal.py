"""
Minimal bpy_lib for MeshCoder visualization.
Extracted from MeshCoder's bpy_lib.py with torch/heavy numpy removed.
Only includes functions needed to execute generated code.
"""

import bpy
import copy
from math import radians, pi, cos, sin, tan, atan2, sqrt, ceil
import math
import bmesh
from mathutils import Vector, Quaternion
import mathutils

# Simple list/array operations to replace numpy where needed
def array_to_list(arr):
    """Convert numpy-like array to list"""
    if isinstance(arr, list):
        return arr
    return list(arr)

# ============================================================================
# Core Functions
# ============================================================================

def delete_all():
    """Delete all objects in the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def recalculate_normals(name, inside=False):
    """Make sure the normal of the mesh is all facing outward or inside"""
    obj = bpy.data.objects[name]
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=inside)
    bpy.ops.object.editmode_toggle()


def weld(name, merge_threshold=1e-3):
    """Apply WELD modifier to the object"""
    bpy.data.objects[name].modifiers.new("Weld", "WELD")
    bpy.data.objects[name].modifiers["Weld"].merge_threshold = merge_threshold
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.modifier_apply(modifier="Weld")


def solidify(name, thickness=0.01, offset=-1, boundary_mode=None):
    """Add thickness to the face"""
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.data.objects[name].select_set(False)
    bpy.data.objects[name].modifiers.new("Solidify", "SOLIDIFY")
    bpy.data.objects[name].modifiers["Solidify"].thickness = thickness
    bpy.data.objects[name].modifiers["Solidify"].offset = offset
    if boundary_mode:
        bpy.data.objects[name].modifiers["Solidify"].solidify_mode = 'NON_MANIFOLD'
        bpy.data.objects[name].modifiers["Solidify"].nonmanifold_boundary_mode = boundary_mode
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.modifier_apply(modifier="Solidify")


def bevel(name, width, segments=8):
    """Add bevel modifier and apply to the object"""
    if width == 0:
        return
    obj = bpy.data.objects.get(name)
    obj.select_set(True)
    bpy.data.objects[name].modifiers.new("Bevel", "BEVEL")
    bpy.data.objects[name].modifiers["Bevel"].width = width
    bpy.data.objects[name].modifiers["Bevel"].segments = segments
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    if bpy.data.objects[name].type == 'MESH':
        bpy.ops.object.modifier_apply(modifier="Bevel")
    elif bpy.data.objects[name].type == 'CURVE':
        bpy.context.object.modifiers["Bevel"].affect = 'VERTICES'
        bpy.ops.object.convert(target='MESH')
        bpy.ops.object.modifier_apply(modifier="Bevel")
        bpy.ops.object.convert(target='CURVE')


# ============================================================================
# Primitive Creation
# ============================================================================

def create_primitive(name, primitive_type="cube", location=None, scale=None, rotation=None, 
                     rotation_mode='QUATERNION', apply=False, x_subdivisions=None, y_subdivisions=None):
    """Create primitive object (simplified - no torch/numpy dependencies)"""
    
    # Create the primitive
    if primitive_type in ["cylinder", "cone", "uv_sphere", "torus"]:
        getattr(bpy.ops.mesh, f"primitive_{primitive_type}_add")()
    elif primitive_type == "cube":
        bpy.ops.mesh.primitive_cube_add()
    elif primitive_type == "grid":
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=x_subdivisions, y_subdivisions=y_subdivisions)
    else:
        getattr(bpy.ops.mesh, f"primitive_{primitive_type}_add")()
    
    primitive = bpy.context.object
    primitive.name = name
    
    # Apply transformations
    if location:
        primitive.location = location
    if scale:
        primitive.scale = scale
    if rotation:
        if rotation_mode == 'XYZ':
            primitive.rotation_euler = [angle * pi for angle in rotation]
        elif rotation_mode == 'QUATERNION':
            primitive.rotation_mode = 'QUATERNION'
            primitive.rotation_quaternion = rotation
    
    if apply:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    return primitive


# ============================================================================
# Curve Creation
# ============================================================================

def create_curve(name, profile_name=None, control_points=[], points_radius=[], handle_type=[], 
                closed=False, center="POINT", thickness=None, fill_caps="none", flip_normals=False,
                bevel_width=None, bevel_segments=8, resolution=24, volumn_origin=True):
    """Creates a curve object"""
    
    if isinstance(name, str):
        type_dict = {0: "AUTO", 1: "VECTOR", 2: "ALIGNED", 3: "FREE"}
        
        control_points_tmp = copy.deepcopy(control_points)
        num_handle_co = handle_type.count(3)
        num_control_points = len(control_points) - num_handle_co
        
        curveData = bpy.data.curves.new(name, type='CURVE')
        curveData.dimensions = '3D'
        
        bezierSpline = curveData.splines.new('BEZIER')
        bezierSpline.bezier_points.add(num_control_points - 1)
        bezierSpline.use_cyclic_u = closed
        
        for i in range(num_control_points):
            bezier_point = bezierSpline.bezier_points[i]
            bezier_point.handle_left_type = type_dict[handle_type[2*i]]
            if type_dict[handle_type[2*i]] == "FREE":
                bezier_point.handle_left = control_points.pop(0)
            bezier_point.co = control_points.pop(0)
            bezier_point.handle_right_type = type_dict[handle_type[2*i+1]]
            if type_dict[handle_type[2*i+1]] == "FREE":
                bezier_point.handle_right = control_points.pop(0)
            bezier_point.radius = points_radius[i] if len(points_radius) != 0 else 1.0
        
        if resolution:
            curveData.resolution_u = resolution
        
        curveOB = bpy.data.objects.new(name, curveData)
        
        if profile_name is not None:
            curveData.bevel_mode = "OBJECT"
            curveData.splines[0].use_smooth = False
            scn = bpy.context.scene.collection
            scn.objects.link(curveOB)
            
            if bevel_width is not None:
                bevel(name=name, width=bevel_width, segments=bevel_segments)
                curveData = bpy.data.objects[name].data
                curveData.bevel_mode = 'OBJECT'
            
            curveData.bevel_object = bpy.data.objects[profile_name]
            curveData.use_fill_caps = (fill_caps == "both")
            curveData.resolution_u = resolution
            
            bpy.context.view_layer.objects.active = bpy.data.objects[name]
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.convert(target='MESH')
            bpy.data.objects.remove(bpy.data.objects[profile_name], do_unlink=True)
            
            if volumn_origin:
                bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
            
            if flip_normals:
                recalculate_normals(name, inside=True)
            else:
                recalculate_normals(name, inside=False)
            
            if thickness and thickness > 1e-10:
                solidify(name, thickness)
            
            weld(name, 1e-5)
            
            return curveOB
        else:
            scn = bpy.context.scene.collection
            scn.objects.link(curveOB)
            if center == "MEDIAN":
                bpy.data.objects[name].select_set(True)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
            return {"name": name, "points": control_points_tmp, "handle_type": handle_type, "closed": closed, "center": center}
    
    elif profile_name is None:
        # Handle list of curves
        if isinstance(profile_name, str):
            profile_name = [profile_name] * len(name)
        if len(points_radius) != 0 and isinstance(points_radius[0], (float, int)):
            points_radius = [points_radius] * len(name)
        elif len(points_radius) == 0:
            points_radius = [[]] * len(name)
        if isinstance(handle_type[0], int):
            handle_type = [handle_type] * len(name)
        if isinstance(closed, bool):
            closed = [closed] * len(name)
        if isinstance(center, str):
            center = [center] * len(name)
        
        for i in range(len(name)):
            create_curve(name=name[i], control_points=control_points[i], points_radius=points_radius[i],
                        handle_type=handle_type[i], closed=closed[i], center=center[i], 
                        thickness=thickness, fill_caps=fill_caps, flip_normals=flip_normals, resolution=resolution)
        
        return {"name": name, "points": control_points, "handle_type": handle_type, "closed": closed, "center": center}


def fill_grid(name, thickness=None, use_interp_simple=True):
    """Fill grid on mesh"""
    obj = bpy.data.objects[name]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    if obj.type != 'MESH':
        if obj.type == 'CURVE':
            # Convert curve to mesh
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.select_set(True)
            bpy.ops.object.convert(target="MESH")
            obj = bpy.context.active_object
            weld(obj.name, merge_threshold=1e-4)
    
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_grid(use_interp_simple=use_interp_simple)
    bpy.ops.object.mode_set(mode="OBJECT")
    
    if thickness is not None:
        solidify(name, thickness)


def bridge_edge_loops(name, profile_name, number_cuts=None, smoothness=0., profile_shape_factor=0., 
                     twist=0, interpolation='PATH', profile_shape='SMOOTH', flip_normals=False, fill_caps="none"):
    """Bridges edge loops between multiple objects"""
    
    if isinstance(name, str):
        assert len(profile_name) >= 2
        assert interpolation in ['LINEAR', 'PATH', 'SURFACE']
        assert profile_shape in ['SMOOTH', 'SPHERE', 'ROOT', 'INVERSE_SQUARE', 'SHARP', 'LINEAR']
        
        # Convert curves to mesh if needed
        for b_name in profile_name:
            if bpy.data.objects.get(b_name).type == 'CURVE':
                bpy.context.view_layer.objects.active = bpy.data.objects[b_name]
                bpy.data.objects[b_name].select_set(True)
                bpy.ops.object.convert(target='MESH')
                bpy.data.objects[b_name].select_set(False)
        
        # Handle fill caps
        if fill_caps == 'start':
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[profile_name[0]].select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects[profile_name[0]]
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.edge_face_add()
            bpy.ops.object.mode_set(mode='OBJECT')
        elif fill_caps == 'end':
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[profile_name[-1]].select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects[profile_name[-1]]
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.edge_face_add()
            bpy.ops.object.mode_set(mode='OBJECT')
        
        scene = bpy.context.scene
        
        # Copy objects
        start_obj = bpy.data.objects[profile_name[0]].copy()
        end_obj = bpy.data.objects[profile_name[-1]].copy()
        start_obj.data = bpy.data.objects[profile_name[0]].data.copy()
        end_obj.data = bpy.data.objects[profile_name[-1]].data.copy()
        scene.collection.objects.link(start_obj)
        scene.collection.objects.link(end_obj)
        profile_name[0] += ".001"
        profile_name[-1] += ".001"
        
        # Select and join
        bpy.ops.object.select_all(action='DESELECT')
        for b_name in profile_name:
            bpy.data.objects[b_name].select_set(True)
        
        if 0. not in bpy.data.objects[profile_name[0]].scale:
            bpy.context.view_layer.objects.active = bpy.data.objects[profile_name[0]]
        else:
            bpy.context.view_layer.objects.active = bpy.data.objects[profile_name[-1]]
        
        if number_cuts is None:
            number_cuts = 32
        
        bpy.ops.object.join()
        bpy.context.object.name = name
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bridge_edge_loops(
            twist_offset=twist,
            number_cuts=number_cuts,
            smoothness=smoothness,
            profile_shape_factor=profile_shape_factor,
            interpolation=interpolation,
            profile_shape=profile_shape
        )
        
        if fill_caps == 'both':
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.edge_face_add()
            bpy.ops.mesh.select_all(action='DESELECT')
        
        bpy.ops.object.mode_set(mode='OBJECT')


def add_simple_deform_modifier(name, mode='BEND', angle=0., origin=[0, 0, 0], rotation=Quaternion([1, 0, 0, 0]), axis='Z'):
    """Add a simple deform modifier to the object"""
    origin = origin.copy() if isinstance(origin, list) else origin
    rotation = rotation.copy() if isinstance(rotation, Quaternion) else Quaternion(rotation)
    
    if isinstance(rotation, list):
        rotation = Quaternion(rotation)
    if isinstance(origin, list) and len(origin) > 0 and isinstance(origin[-1], list):
        origin = origin[-1]
    
    obj = bpy.data.objects.get(name)
    if not obj:
        return
    
    # Create empty object as deform origin
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=origin)
    empty = bpy.context.object
    empty.rotation_mode = 'QUATERNION'
    empty.rotation_quaternion = rotation
    
    # Add modifier
    modifier = obj.modifiers.new(name="SimpleDeform", type='SIMPLE_DEFORM')
    modifier.deform_method = mode
    modifier.deform_axis = axis
    modifier.angle = angle
    modifier.origin = empty
    
    # Apply modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="SimpleDeform")
    
    # Clean up empty
    bpy.data.objects.remove(empty, do_unlink=True)


# ============================================================================
# Helper to create circles (used by some curve operations)
# ============================================================================

def create_circle(name, radius=1.0, center='POINT'):
    """Create a circle curve"""
    bpy.ops.curve.primitive_bezier_circle_add(radius=radius)
    circle = bpy.context.object
    circle.name = name
    if center == 'MEDIAN':
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    return circle

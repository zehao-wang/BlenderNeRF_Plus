import os
import shutil
import bpy
from . import blender_nerf_operator

# def enable_gpu():
#     bpy.context.scene.render.engine = 'CYCLES'
#     bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

#     # List available CUDA devices
#     for device in bpy.context.preferences.addons['cycles'].preferences.devices:
#         print(device.name, ":", device.use)

#     # Activate the CUDA device
#     for device in bpy.context.preferences.addons['cycles'].preferences.devices:
#         if device.type == 'CUDA':
#             device.use = True
#     bpy.context.scene.cycles.device = 'GPU'

# train and test cameras operator class
class TrainTestCameras(blender_nerf_operator.BlenderNeRF_Operator):
    '''Train and Test Cameras Operator'''
    bl_idname = 'object.train_test_cameras'
    bl_label = 'Train and Test Cameras TTC'

    def execute(self, context):
        # enable_gpu()
        scene = context.scene
        train_camera = scene.camera_train_target
        test_camera = scene.camera_test_target

        # check if cameras are selected : next errors depend on existing cameras
        if train_camera == None or test_camera == None:
            self.report({'ERROR'}, 'Be sure to have selected a train and test camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='TTC')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_train_data = self.get_camera_intrinsics(scene, train_camera)
        output_test_data = self.get_camera_intrinsics(scene, test_camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.ttc_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)
            
        # initial properties might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath
        scene.init_frame_end = scene.frame_end

        if scene.test_data:
            # testing transforms
            output_test = os.path.join(output_path, 'test')
            os.makedirs(output_test, exist_ok=True)
            scene.camera = test_camera
            # scene.render.filepath = os.path.join(output_test, '')
            output_test_data['frames'] = self.get_camera_extrinsics(scene, test_camera, mode='TEST', method='TTC')
            self.save_json(output_path, 'transforms_test.json', output_test_data)
            output_test_data['frames'] = self.get_camera_extrinsics_and_render(scene, test_camera, mode='TEST', method='TTC', render_root=output_test)
            self.save_json(output_path, 'transforms_test.json', output_test_data)

            # # rendering
            # if scene.render_frames:
            #     output_test = os.path.join(output_path, 'test')
            #     os.makedirs(output_test, exist_ok=True)
            #     # scene.rendering = (False, True, False)
            #     # scene.frame_end = scene.frame_start + scene.ttc_nb_frames_test - 1
            #     # scene.render.filepath = os.path.join(output_test, '') 
            #     # bpy.ops.render.render(animation=True, write_still=True) 
            #     scene.camera = test_camera
            #     scene.rendering = (True, True, False)
            #     scene.frame_step = scene.ttc_test_step # update frame step
            #     scene.frame_end = scene.frame_start + scene.ttc_nb_frames_test - 1
            #     scene.render.filepath = os.path.join(output_test, '') # training frames path
            #     bpy.ops.render.render(animation=True, write_still=True) # render scene

        if scene.logs: 
            self.save_log_file(scene, output_path, method='TTC')
            self.save_json(output_path, filename='gpu_stat.txt', data={
                "gpu_enabled": bpy.context.scene.cycles.device,
                "engine": bpy.context.scene.render.engine,
                "device_type": bpy.context.preferences.addons['cycles'].preferences.compute_device_type,
                "devices": [
                    (device.name, device.type, device.use) for device in bpy.context.preferences.addons['cycles'].preferences.devices
                ]

            })

        if scene.train_data:
            # training transforms
            output_train = os.path.join(output_path, 'train')
            os.makedirs(output_train, exist_ok=True)
            scene.camera = train_camera
            # scene.render.filepath = os.path.join(output_train, '')
            output_train_data['frames'] = self.get_camera_extrinsics(scene, train_camera, mode='TRAIN', method='TTC')
            self.save_json(output_path, 'transforms_train.json', output_train_data)
            output_train_data['frames'] = self.get_camera_extrinsics_and_render(scene, train_camera, mode='TRAIN', method='TTC',  render_root=output_train)
            self.save_json(output_path, 'transforms_train.json', output_train_data)

            # # rendering
            # if scene.render_frames:
            #     output_train = os.path.join(output_path, 'train')
            #     os.makedirs(output_train, exist_ok=True)
            #     # scene.rendering = (False, True, False)
            #     # scene.frame_end = scene.frame_start + scene.ttc_nb_frames - 1 # update end frame
            #     # scene.render.filepath = os.path.join(output_train, '') # training frames path
            #     # bpy.ops.render.render(animation=True, write_still=True) # render scene
            #     scene.camera = train_camera
            #     scene.rendering = (True, True, False)
            #     scene.frame_step = scene.ttc_train_step # update frame step
            #     scene.frame_end = scene.frame_start + scene.ttc_nb_frames - 1 # update end frame
            #     scene.render.filepath = os.path.join(output_train, '') # training frames path
            #     bpy.ops.render.render(animation=True, write_still=True) # render scene

        # # if frames are rendered, the below code is executed by the handler function
        # if not any(scene.rendering):
        #     # compress dataset and remove folder (only keep zip)
        #     shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
        #     shutil.rmtree(output_path)

        return {'FINISHED'}
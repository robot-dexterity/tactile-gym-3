"""
python launch_servo_control.py -r sim -s tactip -t edge_2d -o circle square
"""
import os
import itertools as it
import time as t
import numpy as np

BASE_DATA_PATH = './tactile_data'

from common.utils import load_json_obj, make_dir
from cri.transforms import inv_transform_euler
# from user_input.slider import Slider

from demos.demo_servo.servo_utils.controller import PIDController
from demos.demo_servo.servo_utils.labelled_model import LabelledModel
from data_collection.collect_data.setup_embodiment import setup_embodiment

# from demos.demo_servo.servo_utils.utils_plots import PlotContour3D as PlotContour
from learning.supervised.image_to_feature.cnn.label_encoder import LabelEncoder
from learning.supervised.image_to_feature.cnn.setup_model import setup_model

from demos.demo_servo.setup_servo import setup_servo, setup_parse


def servo(
    robot,
    sensor,
    pose_model,
    controller,
    image_dir,
    task_params,
    show_plot=False,
    # show_slider=False,
):

    # initialize peripherals
    # if show_plot:
    #     plotContour = PlotContour(robot.coord_frame)
    # if show_slider:
    #     slider = Slider(controller.ref)

    # move to initial pose from 50mm above workframe
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints([*robot.joint_angles[:-1], 0])

    # zero pose and clock
    pose = [0, 0, 0, 0, 0, 0]
    robot.move_linear(pose)

    # turn on servo mode if set
    robot.controller.servo_mode = task_params.get('servo_mode', False)
    robot.controller.time_delay = task_params.get('time_delay', 0.0)

    # timed iteration through servo control
    t_0 = t.time()
    for i in range(task_params['num_iterations']):

        # get current tactile observation
        image_outfile = os.path.join(image_dir, f'image_{i}.png')
        tactile_image = sensor.process(image_outfile)

        # predict pose from observations
        pred_pose = pose_model.predict(tactile_image)

        # servo control output in sensor frame
        servo = controller.update(pred_pose[:6])

        # new pose applies servo in end effector frame
        pose = inv_transform_euler(servo, robot.pose)
        robot.move_linear(pose)

        # optional peripheral: plot trajectory, reference slider
        # if show_plot:
        #     plotContour.update(pose)
        # if show_slider:
        #     controller.ref = slider.read()

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f'\n step {i+1} time {np.array([t.time()-t_0])}: pose: {pose}')

    # finish 50mm above initial pose and zero joint_6
    robot.controller.servo_mode = False
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()

    # optionally save plot and render view
    # if show_plot:
    #     plot_outfile = os.path.join(image_dir, r"../trajectory.png")
    #     plotContour.save(plot_outfile)


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.dataset, args.task, args.model in it.product(args.datasets, args.tasks, args.models):
        for args.object, args.sample_num in zip(args.objects, args.sample_nums):

            run_dir_name = '_'.join(filter(None, [args.object, *args.run_version]))

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, args.task, run_dir_name)
            image_dir = os.path.join(save_dir, "processed_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # load model, environment and image processing parameters
            model_dir = os.path.join(BASE_DATA_PATH, output_dir, args.dataset, args.task, args.model)
            env_params = load_json_obj(os.path.join(model_dir, 'env_params'))
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
            model_label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
            if os.path.isfile(os.path.join(model_dir, 'processed_image_params.json')):
                sensor_image_params = load_json_obj(os.path.join(model_dir, 'processed_image_params'))
            else:
                sensor_image_params = load_json_obj(os.path.join(model_dir, 'sensor_image_params'))

            # setup control and update env parameters from data_dir
            control_params, env_params, task_params = setup_servo(
                args.sample_num,
                args.task,
                args.object,
                args.model,
                env_params,
                save_dir
            )

            # setup the robot and sensor
            robot, sensor = setup_embodiment(
                env_params,
                sensor_image_params
            )

            # setup the controller
            pid_controller = PIDController(**control_params)

            # create the label encoder/decoder
            label_encoder = LabelEncoder(model_label_params, device=args.device)

            # setup the model
            model = setup_model(
                in_dim=model_image_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                saved_model_dir=model_dir,
                device=args.device
            )
            model.eval()

            pose_model = LabelledModel(
                model,
                model_image_params['image_processing'],
                label_encoder,
                device=args.device
            )

            # run the servo control
            servo(
                robot,
                sensor,
                pose_model,
                pid_controller,
                image_dir,
                task_params
            )


if __name__ == "__main__":

    args = setup_parse(
        robot='sim_ur',
        sensor='tactip',
        datasets=['edge_2d_shear'],
        tasks=['servo_2d'],
        models=['simple_cnn'],
        objects=['circle', 'square'],
        sample_nums=[100, 100],
        run_version=[''],
        device='cuda'
    )

    launch(args)

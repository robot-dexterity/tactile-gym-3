import itertools as it
import cv2
import torch
from utils.process_data.process_image_data import process_image_data
from data_collection.setup_collect_data import setup_collect_data
from learning.supervised.image_to_feature.utils.parse_args import parse_args
from learning.supervised.image_to_image.supervised.models import create_model
from learning.supervised.image_to_image.utils.utils_learning import seed_everything
from learning.supervised.image_to_feature.learning.setup_training import setup_training
from learning.supervised.image_to_feature.utils.label_encoder import LabelEncoder
from utils.image_transforms import process_image
from utils.image_transforms import augment_image
import os
import numpy as np
from data_collection.collect_data.setup_embodiment import setup_embodiment
from time import time
import matplotlib.pyplot as plt

import time
import re
from utils.dobot_api import DobotApiDashboard, DobotApiMove, DobotApi

# Robot IP address and ports
ip = "192.168.1.6"  # Please modify the IP address according to the actual situation
dashboard_port = 29999#29999
move_port = 30003#30003
feed_port = 30004

# Create connections to each port of the robot
dashboard = DobotApiDashboard(ip, dashboard_port)
move = DobotApiMove(ip, move_port)
feed = DobotApi(ip, feed_port)

# Enable the robot
dashboard.EnableRobot()

HOME_POS = [334, 10, -50, -54]  # Start position, units: mm, degrees

def load_posenet(args):
    train_data_dirs = [
        "../../tactile_data/data/tactile_sim2real/ur_tactip/edge_2d/train_shear"
    ]
    # setup parameters
    learning_params, model_params, label_params, image_params = setup_training(
        'simple_cnn_mdn_jl',
        'edge_2d',
        train_data_dirs,
        None
    )
    # create the label encoder/decoder and plotter
    label_encoder = LabelEncoder(label_params, args.device)
    # create the model
    seed_everything(learning_params['seed'])
    model = create_model(
        in_dim=image_params['image_processing']['dims'],
        in_channels=1,
        out_dim=label_encoder.out_dim,
        model_params=model_params,
        saved_model_dir="./my_models/mg400_tactip_posenet_mdn/simple_cnn_mdn_jl",
        device=args.device
    )
    model.eval()
    return model.to(args.device), label_encoder

def launch(model, label_encoder, args):
    robot = None

    for args.task, args.input in it.product(args.tasks, args.inputs):
        for args.data_dir, args.sample_num in zip(args.data_dirs, args.sample_nums):
            positions = []

            # setup parameters
            env_params, sensor_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                "../../tactile_data/data/tactile_sim2real/ur_tactip/edge_2d/train_shear"
            )

            learning_params, model_params, label_params, image_params_2 = setup_training(
                'simple_cnn_mdn_jl',
                'edge_2d',
                ["../../tactile_data/data/tactile_sim2real/ur_tactip/edge_2d/train_shear"],
                None
            )

            if robot is None:
                # setup embodiment
                robot, sensor = setup_embodiment(
                    env_params,
                    {"type": "midi",
                     "source": 1,#4,
                     "exposure": -7,
                     "gray": True,
                     "bbox": [110, 0, 550, 440]},
                    sim_sensor=False
                )
            print("ROBOT EMBODIED!")

            if "sim" in args.robot:
                rest_pos = (0, 0, -50, 0, 0, 0)
            else:
                rest_pos = (0, 0, 50, 0, 0, 0)
            rest_z = rest_pos[2]
            # move.ServoP(HOME_POS[0], HOME_POS[1], HOME_POS[2], HOME_POS[3], 3)
            robpose = [float(i) for i in re.search(r'\{(.*?)\}', dashboard.GetPose()).group(1).split(",")]
            move.MovL(robpose[0], robpose[1], robpose[2], 180)
            input("press enter...")
            # time.sleep(3)
            #robot.move_linear(rest_pos)

            # collect reference image
            image_outfile = os.path.join('./img.png')

            first = True
            NEWP_DEL = np.array([0, 50, 50, 0, 0, 0])
            start_time = time.time()
            plot_this = []
            while time.time() - start_time < 100:
                print("STARTING")
                _ = sensor.process(image_outfile)
                img = cv2.imread('./img.png')
                image_params = {
                    # "bbox": (12, 12, 240, 240) #sim

                    "bbox": (25, 90, 372, 441), #tactip
                    "thresh": [501, 1] #tactip
                }
                image = process_image(img, **image_params)
                image = np.repeat(image, 3, 2)
                # cv2.imwrite('./img.png', image)
                # image2 = cv2.imread('./img.png')
                # preprocess/augment image
                processed_image = process_image(
                    image,
                    gray=True,
                    bbox=image_params_2['image_processing']['bbox'],
                    dims=(128, 128),
                    stdiz=image_params_2['image_processing']['stdiz'],
                    normlz=image_params_2['image_processing']['normlz'],
                    thresh=image_params_2['image_processing']['thresh'],
                )
                processed_image = augment_image(
                    processed_image,
                    rshift=image_params_2['augmentation']['rshift'],
                    rzoom=image_params_2['augmentation']['rzoom'],
                    brightlims=image_params_2['augmentation']['brightlims'],
                    noise_var=image_params_2['augmentation']['noise_var']
                )
                # put the channel into first axis because pytorch
                processed_image = np.rollaxis(processed_image, 2, 0)

                # mod_output = model(torch.tensor(processed_image).to(args.device).unsqueeze(0))
                # shear = mod_output.detach().cpu().numpy()

                inp = torch.tensor(processed_image).to(args.device).unsqueeze(0)
                mod_output = model(inp)
                mod_output = mod_output[1].squeeze(0)
                print(mod_output)
                predictions_dict = label_encoder.decode_label(mod_output)

                pose_z, shear_x, shear_y, shear_Rz = (predictions_dict['pose_z'].item(),
                                                      predictions_dict['shear_x'].item(),
                                                      predictions_dict['shear_y'].item(),
                                                      predictions_dict['shear_Rz'].item())


                robpose = [float(i) for i in re.search(r'\{(.*?)\}', dashboard.GetPose()).group(1).split(",")]

                positions.append(robpose)

                shear_x_true = shear_x * np.cos(robpose[3]) + shear_y * np.sin(robpose[3])
                shear_y_true = -shear_x * np.sin(robpose[3]) + shear_y * np.cos(robpose[3])

                # if pose_z > 4.25:
                #     shear_x = 0
                #     shear_y = 0
                # if np.abs(pose_z) < 1:
                #     pose_z = 0
                # if np.abs(shear_x) < 0.05: shear_x = 0
                # if np.abs(shear_y) < 0.05: shear_y = 0

                if pose_z > 5:
                    shear_x_true = 0
                    shear_y_true = 0
                pose_diff = np.array([shear_x_true, shear_y_true, (4.05-pose_z),
                                      # 0,0,0,
                                      #0,#shear_Rz,
                                      0, 0, 0])#, shear_Rz])
                pose_diff = pose_diff/20
                #pose_diff = pose_diff * np.array([7,7,0,0,0,0])
                new_pose = robpose + pose_diff#np.array([10,10,5,1,1,5]))
                if np.linalg.norm(pose_diff) > 0:
                    print("MOVING")
                    # move.ServoP(new_pose[0], new_pose[1], new_pose[2], new_pose[3], 0.02)
                    move.ServoP(new_pose[0], new_pose[1], new_pose[2],
                                new_pose[3],
                                0.02)
                    # move.MovJExt(new_pose[5], 0.02)
                NEWP_DEL = NEWP_DEL*np.array([1,-1,1,1,1,1])
                if first:
                    first = False
                print(f"{shear_x}, {shear_y}, {pose_z}")
                plot_this.append([time.time(), new_pose[0], new_pose[1], new_pose[2], new_pose[5]])
            plot_this = np.array(plot_this)

            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the points
            ax.plot(np.array(plot_this)[:, 1],
                    np.array(plot_this)[:, 2],
                    np.array(plot_this)[:, 3],
                    marker='o', label='Robot Arm Movement')

            # Customize the plot
            ax.set_title('3D Robot Arm Movement')
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            ax.legend()

            # Show the plot
            plt.show()
            positions = np.array(positions)
            np.save('hold_sinusoid.npy', positions)


def process_images(path, args, image_params):
    process_image_data(path, args, image_params)

if __name__ == "__main__":

    args = parse_args(
        inputs=['mg400_tactip'],
        # robot='sim_mg400',
        robot="mg400",
        sensor='tactip',
        tasks=['edge_2d'],
        data_dirs=['train_shear', 'val_shear'],
        # sample_nums=[10],
        device="cpu"
    )

    model, label_encoder = load_posenet(args)
    launch(model, label_encoder, args)
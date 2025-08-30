import argparse
import glob
import os
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from alg.coea.structure import Structure
from analysis.analysis_utils import get_robot_save_path, get_top_robot_ids
from evaluate import evaluate
from utils import load_args


def write(experiment_dir: str, generations: List[int]):
    """
    Generates and saves evaluation videos for the top-performing robots across specified generations.

    Side Effects:
        - Creates directories for saving videos if they do not exist.
        - Writes evaluation videos as .mp4 files to the output directory.

    Args:
        experiment_dir (str): Path to the experiment directory containing robot data and metadata.
        generations (List[int]): List of generation numbers for which to generate evaluation videos.
    """

    # Load env name
    coea_args = load_args(os.path.join(experiment_dir, "metadata"))
    env_name = coea_args.env_name

    csv_path_1 = os.path.join(experiment_dir, "robot_1", "fitnesses.csv")
    csv_path_2 = os.path.join(experiment_dir, "robot_2", "fitnesses.csv")

    for generation in generations:

        # Get top 4 robot ids
        top_n_id_1 = get_top_robot_ids(csv_path_1, generation=generation, top_n=4)
        top_n_id_2 = get_top_robot_ids(csv_path_2, generation=generation, top_n=4)

        for r, (id1, id2) in enumerate(zip(top_n_id_1, top_n_id_2)):

            # Load robot structures
            save_path_1 = get_robot_save_path(os.path.join(experiment_dir, "robot_1"), id1, generation)
            save_path_2 = get_robot_save_path(os.path.join(experiment_dir, "robot_2"), id2, generation)

            structures = {
                "robot_1": Structure.from_save_path(save_path_1),
                "robot_2": Structure.from_save_path(save_path_2),
            }

            # Write video
            print(f"writing {save_path_1} and {save_path_2} ({r+1+4*generation}/{len(generations)*4})")
            movie_dir = os.path.join("analysis", "movies", env_name, "evolution_path")
            os.makedirs(movie_dir, exist_ok=True)
            evaluate(
                structures,
                env_name,
                num_processes=1,
                device=torch.device("cpu"),
                min_num_episodes=1,
                render_mode="rgb_array",
                movie_path=os.path.join(movie_dir, f"{os.path.basename(experiment_dir)}_gen{generation}_top{r}.mp4"),
            )


def concatenate(experiment_dir: str, generations: List[int]):
    """
    Concatenates individual evaluation videos of top-performing robots into a single tiled video for each specified generation.

    Args:
        experiment_dir (str): Path to the experiment directory containing robot data and metadata.
        generations (List[int]): List of generation numbers for which to generate tiled videos.
    """

    # Load env name
    coea_args = load_args(os.path.join(experiment_dir, "metadata"))
    env_name = coea_args.env_name

    video_dir = os.path.join("analysis", "movies", env_name, "evolution_path")

    # Video config
    fps = 50
    frame_size = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    tiled_video_path = os.path.join("analysis", "movies", env_name, f"{os.path.basename(experiment_dir)}_tiled.mp4")
    print(f"Creating tiled video: {tiled_video_path}")

    for generation in tqdm(generations, desc="Processing generations"):

        video_path_list = sorted(glob.glob(os.path.join(video_dir, f"*_gen{generation}_top*.mp4")))

        # Find minimum n frames across videos
        min_n_frames = float("inf")
        for video_path in video_path_list:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            min_n_frames = min(min_n_frames, frame_count)
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()

        # Read frames
        tiled_frames = []
        for video_path in tqdm(video_path_list, desc="Reading frames"):
            cap = cv2.VideoCapture(video_path)
            frames = []
            for _ in range(min_n_frames):
                _ret, frame = cap.read()
                frames.append(frame)
            cap.release()
            tiled_frames.append(frames)

        # Write concatenated frames
        for n in tqdm(range(min_n_frames), desc="Writing frames"):

            frame1, frame2, frame3, frame4 = (
                tiled_frames[0][n],
                tiled_frames[1][n],
                tiled_frames[2][n],
                tiled_frames[3][n],
            )
            frame1 = cv2.resize(frame1, (frame_size[0] // 2, frame_size[1] // 2))
            frame2 = cv2.resize(frame2, (frame_size[0] // 2, frame_size[1] // 2))
            frame3 = cv2.resize(frame3, (frame_size[0] // 2, frame_size[1] // 2))
            frame4 = cv2.resize(frame4, (frame_size[0] // 2, frame_size[1] // 2))

            top_row = np.hstack((frame1, frame2))
            bottom_row = np.hstack((frame3, frame4))
            tiled_frame = np.vstack((top_row, bottom_row))

            # Add lines to separate the frames
            line_color = (0, 0, 0)  # Black color for the lines
            line_thickness = 2

            # Horizontal line
            cv2.line(
                tiled_frame, (0, frame1.shape[0]), (tiled_frame.shape[1], frame1.shape[0]), line_color, line_thickness
            )
            # Vertical line
            cv2.line(
                tiled_frame, (frame1.shape[1], 0), (frame1.shape[1], tiled_frame.shape[0]), line_color, line_thickness
            )

            # Add generation text to the top-right corner
            text = f"generation {generation}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 0, 0)  # Black color for the text
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = tiled_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
            text_y = text_size[1] + 10  # 10 pixels from the top edge
            cv2.putText(tiled_frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

            if writer is None:
                writer = cv2.VideoWriter(tiled_video_path, fourcc, fps, (tiled_frame.shape[1], tiled_frame.shape[0]))

            writer.write(tiled_frame)

    if writer is not None:
        writer.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument(
        "--mode",
        choices=["write", "concatenate"],
        default="write",
        help="Mode: write individual videos or concatenate them",
    )
    parser.add_argument(
        "--generations", type=int, nargs="+", default=[0, 7, 14], help="List of generations to process"
    )

    args = parser.parse_args()

    if args.mode == "write":
        write(args.experiment_dir, args.generations)
    elif args.mode == "concatenate":
        concatenate(args.experiment_dir, args.generations)

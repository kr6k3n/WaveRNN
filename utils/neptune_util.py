import os
import neptune as nt
from utils import zip_util

COLAB_PATH = "/content/WaveRNN/"

EXPERIMENT_NAME = 'training'
CHECKPOINT_DIR = COLAB_PATH + 'checkpoints'


def get_experiment():
    #pylint: disable=global-statement
    global project
    if project is None:
        raise Exception("Global variable project not initalized")

    # pylint: disable=protected-access
    return project._get_current_experiment()


def init_experiment():
    return nt.create_experiment(name=EXPERIMENT_NAME)


def resume_experiment():
    session = nt.sessions.Session()
    project = session.get_project(project_qualified_name='kr6k3n/TTS')
    return project.get_experiments()[-1]  #  get last experiment


def save_current_state_to_neptune(neptune):
    for path in zip_util.get_all_file_paths(CHECKPOINT_DIR):
        neptune.send_artifact(path, path.replace(COLAB_PATH, ""))


def get_checkpoint_from_neptune():
    resume_experiment().download_artifacts("checkpoints", COLAB_PATH )
    zip_util.decompress_filepath(CHECKPOINT_DIR+".zip")


nt.init(project_qualified_name="kr6k3n/TTS")

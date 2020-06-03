import os
import neptune
import zip_util

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
    neptune.create_experiment(name=EXPERIMENT_NAME)


def save_current_state_to_neptune():
    neptune.delete_artifacts("checkpoints.zip")
    os.remove(CHECKPOINT_DIR)
    zip_util.compress_filepath(CHECKPOINT_DIR)
    neptune.send_artifact(CHECKPOINT_DIR+".zip", "checkpoints.zip")


def get_checkpoint_from_neptune():
    get_experiment().download_artifact("checkpoints.zip", CHECKPOINT_DIR+".zip")
    zip_util.decompress_filepath(CHECKPOINT_DIR+".zip")


neptune.set_project('kr6k3n/TTS')
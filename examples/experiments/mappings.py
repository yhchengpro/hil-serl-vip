from experiments.open_drawer.config import TrainConfig as OpenDrawerTrainConfig
from experiments.grasp_bread.config import TrainConfig as GraspBreadTrainingConfig
from experiments.move_bread.config import TrainConfig as MoveBreadTrainConfig
from experiments.press_switch.config import TrainConfig as PressSwitchTrainConfig
from experiments.grasp_bread.config import TrainConfig

CONFIG_MAPPING = {
                "open_drawer": OpenDrawerTrainConfig,
                "grasp_bread": GraspBreadTrainingConfig,
                "move_bread": MoveBreadTrainConfig,
                "press_switch": PressSwitchTrainConfig,
               }
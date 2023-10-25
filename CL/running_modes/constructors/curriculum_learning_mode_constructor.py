import sys 
sys.path.append("../..")
import time  
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel

from running_modes.automated_curriculum_learning.automated_curriculum_runner import AutomatedCurriculumRunner
from running_modes.automated_curriculum_learning.logging import AutoCLLogger
from running_modes.configurations import GeneralConfigurationEnvelope
from running_modes.configurations.automated_curriculum_learning.automated_curriculum_learning_input_configuration import \
    AutomatedCurriculumLearningInputConfiguration
from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.curriculum_learning.curriculum_runner import CurriculumRunner
from running_modes.enums.curriculum_type_enum import CurriculumTypeEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.utils.general import set_default_device_cuda

# import sys 
sys.path.append("../../..") 
from parameters.constants import constants as C
import torch


class CurriculumLearningModeConstructor:
    def __init__(self, configuration,constants=C):

        self.start_time = time.time()

        self.C = constants
        self.configuration = configuration
        

        # create placeholders
        self.agent_model = None
        self.prior_model = None
        self.optimizer = None

        self.current_epoch = None
        self.restart_epoch = None
        self.ts_properties = None

        self.tensorboard_writer = None

        self.n_subgraphs = None ##

        self.best_avg_score = 0

        
    def runner(self):
        self._configuration = self.configuration
        cl_enum = CurriculumTypeEnum

        base_config = BaseConfiguration.parse_obj(self._configuration.parameters)

        if base_config.curriculum_type == cl_enum.MANUAL:
            set_default_device_cuda()
            runner = CurriculumRunner(self._configuration)
        elif base_config.curriculum_type == cl_enum.AUTOMATED:
            
            runner = self._create_automated_curriculum(configuration = self._configuration)
            
        else:
            raise KeyError(f"Incorrect curriculum type: `{base_config.curriculum_type}` provided")

        return runner
    
    def get_ts_properties(self):
        """ Loads the training sets properties from CSV as a dictionary, properties
        are used later for model evaluation.
        """
        filename = self.C.training_set[:-3] + "csv"
        self.ts_properties = util.load_ts_properties(csv_path=filename)

    def define_model_and_optimizer(self):
        """ Defines the model (`self.model`) and the optimizer (`self.optimizer`).
        """
        print("* Defining model and optimizer.", flush=True)
        job_dir = self.C.job_dir
        model_dir = self.C.dataset_dir

        if self.C.restart:
            print("-- Loading model from previous saved state.", flush=True)
            self.restart_epoch = util.get_restart_epoch()
            self.agent_model = torch.load(f"{job_dir}model_restart_{self.restart_epoch}.pth")
            self.prior_model = torch.load(f"{model_dir}model_restart_30.pth")
            self.prev_model = torch.load(f"{model_dir}model_restart_30.pth")
            # Load sklearn activity model
            with open(self.C.data_path + "qsar_model.pickle", 'rb') as file:
                model_dict = pickle.load(file)                                      
                self.drd2_model = model_dict["classifier_sv"]
            

            print(
                f"-- Backing up as "
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth.",
                flush=True,
            )
            shutil.copyfile(
                f"{job_dir}model_restart_{self.restart_epoch}.pth",
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth",
            )

        else:
            print("-- Initializing model from scratch.", flush=True)
            self.agent_model = torch.load(f"{model_dir}model_restart_30.pth")
            self.prior_model = torch.load(f"{model_dir}model_restart_30.pth")
            self.prev_model = torch.load(f"{model_dir}model_restart_30.pth")
            
            # 

            self.restart_epoch = 0

        start_epoch = self.restart_epoch + 1
        end_epoch = start_epoch + self.C.epochs

        print("-- Defining optimizer.", flush=True)
        self.optimizer = torch.optim.Adam(
            params=self.agent_model.parameters(),
            lr=self.C.init_lr,
            weight_decay=self.C.weight_decay,
        )

        print("-- Defining scheduler.", flush=True)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr= self.C.max_rel_lr * self.C.init_lr,
            div_factor= 1. / self.C.max_rel_lr,
            final_div_factor = 1. / self.C.min_rel_lr,
            pct_start = 0.05,
            total_steps=self.C.epochs,
            epochs=self.C.epochs
        )

        return start_epoch, end_epoch   


    def new(self, configuration: GeneralConfigurationEnvelope) -> BaseRunningMode:
        
        self._configuration = configuration
        cl_enum = CurriculumTypeEnum

        base_config = BaseConfiguration.parse_obj(self._configuration.parameters)

        if base_config.curriculum_type == cl_enum.MANUAL:
            set_default_device_cuda()
            runner = CurriculumRunner(self._configuration)
        elif base_config.curriculum_type == cl_enum.AUTOMATED:
            
            runner = self._create_automated_curriculum(configuration = self._configuration)
            
        else:
            raise KeyError(f"Incorrect curriculum type: `{base_config.curriculum_type}` provided")

        return runner

    # @staticmethod()
    def _create_automated_curriculum(self,configuration):
        model_type = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()
        start_epoch, end_epoch = self.define_model_and_optimizer()

        if model_type.DEFAULT == configuration.model_type:
            set_default_device_cuda()
            config = AutomatedCurriculumLearningInputConfiguration.parse_obj(configuration.parameters)
        elif model_type.LINK_INVENT == configuration.model_type:
            set_default_device_cuda()
            config = AutomatedCurriculumLearningInputConfiguration.parse_obj(configuration.parameters)
        else:
            raise KeyError(f"Incorrect model type: `{configuration.model_type}` provided")

        _logger = AutoCLLogger(configuration)
        prior_config = ModelConfiguration(configuration.model_type, model_regime.INFERENCE, config.prior)
        agent_config = ModelConfiguration(configuration.model_type, model_regime.TRAINING, config.agent)
        _prior = self.prior_model
        _agent = self.agent_model
        
        runner = AutomatedCurriculumRunner(config, _logger, _prior, _agent)
        return runner
    
    
        

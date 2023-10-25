# import curriculum_learning_mode_constructor.CurriculumLearningModeConstructor as C
import sys 
sys.path.append("../../..") 
# from parameters.constants import constants as Ca
from curriculum_learning_mode_constructor import CurriculumLearningModeConstructor as C
configuration = 1
a = C(configuration=configuration)
a.define_model_and_optimizer()

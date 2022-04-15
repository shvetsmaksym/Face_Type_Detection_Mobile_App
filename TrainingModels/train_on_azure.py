from azureml.core import Experiment, Run, Workspace, Dataset
import azureml.core

subscription_id = 'ce4bb59f-e69f-4a6a-ac35-620c0e13b5e8'
resource_group = 'ML_AI'
workspace_name = 'ml_for_face_type_detection'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='face_types')
dataset.download(target_path='.', overwrite=False)
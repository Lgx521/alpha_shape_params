import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_directory = os.path.join(project_root, 'checkpoints')
print(save_directory)
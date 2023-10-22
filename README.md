# detect_sleep_states
The aim of this project is to detect sleep onset and wake. We will develop a model trained on wrist-worn accelerometer data in order to determine a person's sleep state

## To start virtual environment:

pip install virtualenv 

# To make virtual environement and name it virtual_env
python -m virtualenv virtual_env

# To activate the vritual_env u just made
.\virtual_env\Scripts\activate

Also, click to an empty area and press "ctrl shift P' -> Select python interpreter and select your virtual environment (virtual_env in this case)

# Pip install the requirements text
pip install -r requirements.txt

# install this virtual_env on onto jupyter
ipython kernel install --user --name=virtual_env

## To generate/update a requirements.txt file after installing some packages for model building:
pip freeze > requirements.txt

 and commit.

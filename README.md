# lao-pddlgym
Implementation of LAO*/ILAO* [[Hansen and Zilberstein, 2001](https://www.sciencedirect.com/science/article/pii/S0004370201001060)] algorithms to solve MDPs described as PDDLGym environments

## Notes
This implementation uses the function `get_successor_states` imported from PDDLGym's `core` module (`pddlgym.core`).
Since this feature is currently available in PDDLGym's repository but not in its latest pypi release as of now,
to use it you'll need to either clone the repository and install it locally or install it via pip by pointing to the repository.
You can do the former by settting up a virtual env ([see here](https://github.com/tomsilver/pddlgym#installing-from-source-if-you-want-to-make-changes-to-pddlgym)) or the latter by running the following:

`$ pip install git+https://github.com/tomsilver/pddlgym`

## Usage
For usage instructions, run `python main.py --help` in the repository's root folder

The following command example can be used to solve the first available instance of the Triangle Tireworld [[Little et al., 2007](http://users.cecs.anu.edu.au/~iain/icaps07.pdf)] environment using the ILAO algorithm and output some results:

`$ python src/main.py --env PDDLEnvTireworld-v0 --problem_index 0 --algorithm ilao --render_and_save`

To simulate an episode after the optimal policy has been found and render each encountered state:

`$ python src/main.py --env PDDLEnvTireworld-v0 --problem_index 0 --algorithm ilao --simulate --render_and_save`

Right now only the heuristic function h(s) = 0 is available, so the performance might not be great.

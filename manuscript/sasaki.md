# 論文の原稿

## 2023.5.19 作成

基本的には、ReadMeを作るイメージ。

This library is a toolbox written in Python language and, therefore, requires Python version 3.9 or later to be installed on the user's environment.
The toolbox employed some packages widely used in a variety kind of numerical analysis fields such as science or industry, named POT (Python Optimal Transport), PyTorch, NumPy, Scipy, Pandas, Scikit-learn, Matplotlib, Seaborn, PyMysql (optional), and Optuna.
The main feature of this toolbox is to help users to find the optimal $\varepsilon$ value in the computation of entropic Gromov-Wasserstain (GW alignment) implemented in POT with Optuna.

This toolbox mainly consisted of three folders, `src`, `utils (this can be seen in the src folder)`, and `script`. In `src` folder in the toolbox, there are some files in which we implemented the functions for the computation of GW alignment, `utils` provides some helper functions to work `src`'s computation smoothly (for example, changing the format of data from NumPy to PyTorch when using GPU to accelerate the computation), and `script` provides some notebook or python files as the tutorial on how to use the function of this toolbox to assist one to understand what they need to do with their data.

We provide two ways for users to compute it. Firstly, we designed some classes to facilitate users to run the optimization who are not well experienced with Python programming but are interested in the analysis of GW alignment. These functions are implemented in `align_representations.py` in `src` and its tutorial is `tutorial_align_representations.py` in `script`. Secondly, we provide an orthodox way to use our toolbox. This will help one to understand when building the code from scratch. The tutorial for this is `tutorial.ipynb` in the `script` folder.

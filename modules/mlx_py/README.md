mlx_py

module that launches monolix from python to simulate tumor growth.

______________________________________

Library installation (MlxConnectors):
MlxConnectors is the Monolix library written in R (references: http://monolix.lixoft.com/monolix-api/)
To install it (on mac OS):
- first install the version of Monolix MonolixSuite2018R1 from http://lixoft.com/downloads/
- write the following R commands in R environment:  
 ``>install.packages('R6')``  
 ``>install.packages('RJSONIO')``  
 ``>install.packages("/Applications/MonolixSuite2018R1.app/Contents/Resources/mlxsuite/mlxConnectors/R/MlxConnectors.tar.gz", repos = NULL, type='source')``

______________________________________

Python packages:
- rpy2 (python interface to the R language)
- matplotlib
- pandas
- numpy
- os
- sys

______________________________________

How to import the module:

from mlx_py import *

So that the following functions are imported:
- launch_monolix: performs the Monolix simulation
- save_monolix_graphics: saves the graphics relative to the monolix simulation
- save_results: compares different simulations
Type 'help(name_of_the_function)' on python to see the inputs and the outputs of the functions

______________________________________
Currently, there is an issue when installing rpy2 using pip3, with R older than 3.3.0. The following seems to resolve rpy2 installation

>brew install --with-toolchain llvm  
export PATH="/usr/local/opt/llvm/bin:$PATH"  
export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"  
pip3 install rpy2

(see https://bitbucket.org/rpy2/rpy2/issues/403/cannot-pip-install-rpy2-with-latest-r-340)

However, running launch_monolix is still not functional...

______________________________________
DATA FILE

The data file must be compatible with Monolix. It has to be a text file (.txt, .dat or .csv) and it must have three columns named 'ID', 'Time', 'Observation' separated with space or tab. The last line of the file must be an empty row. No extra space has to be inserted at the beginning or at the end of each row.

______________________________________
Issue when not working on SB machine resolved by doing:
conda create -n py36 python=3.6 numpy matplotlib rpy2 jupyter pandas       # this creates the environment
source activate py36							                                           # this activates the environment
jupyter notebook

To deactivate the environment
source deactivate

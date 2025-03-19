.. flowcam documentation master file, created by
   sphinx-quickstart on Mon Mar 13 14:12:52 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Probabalistic modeling of Flowcam data
=======================================
This project shows how to perform probabilistic modeling for tabular data.
To this end, we employ one decision tree based
approach ngboost_ and two neural network (NN)
based approaches. The first NN based approach is aleoteric. This method
qunatifies uncertainty in data generating process. To record aleoteric
uncertainty in a NN, we have to modify the output layer. In this case, the output
layer is a probability distribution instead of a fully connected layer. The
parameters of this distribution are learned during model training. The
other kind of uncertainty is epistemic uncertainty. This method quantifies
uncertainty in our knowledge. It should be noted that recording epistemic
uncertainty increases the learnable parameters of NN far more than vanilla
or aleoteric NN. However, the good news is that this type of uncertainty
can be reduced by
collecting more data. Such NNs are also known as bayesian NNs.
In such NNs, the weights (and biases) are distributions instead of scaler
values. The parameter of these distributions are again learned during model
training. We can also combine both aleoteric and epistemic uncertainties to make
our NNs into a probablistic NN. Such a NNs have weights as distributions and
output layer as distribution as well.

Data
-----
The input data consists of six parameters affecting the removal of microbes from
wastewater such as wastewater concentration (initial cell count), solution pH, amount of ultravoilet
radiation with which the wastewater sample was treated, the time
at which the sample was treated etc. We have two targets, disinfection efficiency (%) and area.
The disinfection efficiency is the measure of decrease in microbial cell counts after treating
the sample. We
build separate models of each category. Each model receives same inputs but predicts
different target. A comprehensive analysis of data is given in
:ref:`sphx_glr_auto_examples_eda.py`

Reproducibility
----------------
To replicate the experiments, you need to install all requirements given in
`requirements file <https://gitlab.com/atrcheema/flowcam/-/blob/main/requirements.txt>`_ .
If your results are quite different from what are presented here, then make sure
that you are using the exact versions of the libraries which were used
at the time of running of these scripts. These versions are given printed
at the start of each script. Download all the .py files in the scripts including
utils.py (:ref:`sphx_glr_auto_examples_utils.py`) file. The data is expected to be
in the ``data`` folder under the scripts folder.


.. toctree::
   :maxdepth: 4

   auto_examples/index



.. _ngboost:
    https://stanfordmlgroup.github.io/projects/ngboost/

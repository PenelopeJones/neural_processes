# Neural Processes
A neural process is an approximate representation of a stochastic process, modelled using a neural network architecture. 

When we train a neural process, we are effectively learning
an approximation to some underlying stochastic process prior distribution. 

Neural processes can be useful as alternatives to Gaussian processes, 
especially when
- The stochastic process prior is not _obvious_.
- The number of data points and datasets is large.

This repository contains code to train three members of the neural process family:
 - Conditional Neural Processes [\[1\]](https://arxiv.org/abs/1807.01613)
 - Neural Processes [\[2\]](https://arxiv.org/abs/1807.01622)
 - Attentive Neural Processes [\[3\]](https://arxiv.org/abs/1901.05761)
 
It also contains code to train the model implemented in [\[4\]](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00768?src=recsys),
and my own, neural process inspired, models for the imputation of chemical data. 
 
 
## Experiments
- Regression metalearning task _(1-dimensional toy data)_.
- Imputation metalearning task _(chemical data)_.
 
## Requirements
This code was implemented using Python 3.8.5 and the following packages:
- numpy (1.19.1)
- pandas (1.1.0)
- torch (1.6.0)
- torchvision (0.7.0)
- scikit-learn (0.23.2)
- matplotlib (3.3.0)
- scipy (1.5.2)

## Contact / Acknowledgements
If you use this code for your research, please acknowledge the author (Penelope K. Jones, [pj321@cam.ac.uk](mailto:pj321@cam.ac.uk)). 
Please feel free to contact me if you have any questions about this work.

## References
[\[1\]](https://arxiv.org/abs/1807.01613) Conditional Neural Processes: Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami. ICML 2018.

[\[2\]](https://arxiv.org/abs/1807.01622) Neural Processes: Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh. ICML 2018. 

[\[3\]](https://arxiv.org/abs/1901.05761) Attentive Neural Processes: Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, 
Ali Eslami, Dan Rosenbaum, Oriol Vinyals, Yee Whye Teh. ICLR 2019.

[\[4\]](https://pubs.acs.org/doi/full/10.1021/acs.jcim.8b00768?src=recsys) Imputation of Assay Bioactivity Data Using Deep Learning:
T. M. Whitehead, B. W. J. Irwin, P. Hunt, M. D. Segall, and G. J. Conduit. ACS 2019.

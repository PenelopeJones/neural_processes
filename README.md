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
 
 
## Experiments
- Regression metalearning task _(1-dimensional toy data)_.
- Imputation metalearning task _(chemical data)_.
 
## Requirements
This code was implemented using Python 3.7.6 and the following packages:
- numpy (1.18.3)
- pandas (1.0.3)
- torch (1.5.0)
- torchvision (0.6.0)

## Contact / Acknowledgements
If you use this code for your research, please acknowledge the author (Penelope K. Jones, [pj321@cam.ac.uk](mailto:pj321@cam.ac.uk)). 
Please feel free to contact me if you have any questions about this work.

## References
[\[1\]](https://arxiv.org/abs/1807.01613) Conditional Neural Processes: Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami. ICML 2018.

[\[2\]](https://arxiv.org/abs/1807.01622) Neural Processes: Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S.M. Ali Eslami, Yee Whye Teh. ICML 2018. 

[\[3\]](https://arxiv.org/abs/1901.05761) Attentive Neural Processes: Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, 
Ali Eslami, Dan Rosenbaum, Oriol Vinyals, Yee Whye Teh. ICLR 2019.

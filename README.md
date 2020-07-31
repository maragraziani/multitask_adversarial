# Combining Multi-task and Adversarial Losses
Repository for the paper "Controlling CNNs by combining multitask and adversarial losses" by Mara Graziani, Sebastian Otalora, Henning Muller and Vincent Andrearczyk.

In this paper, a framework for controllable and interpretable training is built on top of successful existing techniques of hard parameter sharing, with the main goal of introducing expert knowledge in the training objectives. 

The learning process is guided by identifying concepts that are relevant or misleading for the task. Relevant concepts are encouraged to appear in the representation through multi-task learning. Undesired and misleading concepts are discouraged by a gradient reversal operation. In this way, a shift in the deep representations can be corrected to match the clinicians' assumptions. 

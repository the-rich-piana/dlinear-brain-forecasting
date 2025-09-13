Out of distribution testing on passive behavioral data. To assess generalization ability and quantify model performance on a fundamentally different cortical state of attention, we repeat experiment 1. But only on the 7000 neuron dataset.

All experiments were performed on one 75.2 minute recording session of a mouse performing the IBL task. The first 61.1 minutes of this recording session contained the mouse performing some 521 trials of the biasedChoiceWorld task. Passive imaging protocol takes place for 14.1 minutes immediately after the end of the active task-behavior protocol. The mouse is shown a Zebra noise stimulus, the wheel is locked, and the mouse no longer receives a water reward, altering its cortical state significantly.

These experiments use the BEHAVIORAL data loader:
class Dataset_Activity_Behavioral(Dataset_Activity) in /cs/student/msc/aibh/2024/gcosta/DLinear/data_provider/data_loader.py

And uses the results from this training script:
/cs/student/msc/aibh/2024/gcosta/DLinear/scripts/EXP_Activity_Behavioral/activity_long_behavioral_7000_neurons_parallel.sh

As well as the results from this active task protocol one (same as Exp_1, but we only take the 7000 neuron results and the use Context Length=48 Prediction length=16 ):
/cs/student/msc/aibh/2024/gcosta/DLinear/scripts/EXP_Activity/activity_long_70_700_7000_neurons_parallel.sh

# Data set \<self-paced\>

Data set provided by Fraunhofer-FIRST, Intelligent Data Analysis Group (Klaus-Robert Müller), and Freie Universität Berlin, Department of Neurology, Neurophysics Group (Gabriel Curio)

Correspondence to Benjamin Blankertz &lt;benjamin.blankertz@tu-berlin.de&gt;

This dataset was recorded from a normal subject during a no-feedback session. The subject sat in a normal chair, relaxed arms resting on the table, fingers in the standard typing position at the computer keyboard. The task was to press with the index and little fingers the corresponding keys in a self-chosen order and timing 'self-paced key typing'. The experiment consisted of 3 sessions of 6 minutes each. All sessions were conducted on the same day with some minutes break inbetween. Typing was done at an average speed of 1 key per second.

## Format of the data

Given are 416 epochs of 500 ms length each ending 130 ms before a keypress. 316 epochs are labeled (0 for upcoming left hand movements and 1 for upcoming right hand movements), the remaining 100 epoches are unlabeled for competition purpose.

Data are provided in the original 1000 Hz sampling and in a version downsampled at 100 Hz (recommended). Files are provided in

1. **Matlab** format (*.mat) containing variables: clab: electrode labels, x_train: training trials (time x channels x trials), y_train: corresponding labels (0: left, 1: right), x_test: test trials (time x channels x trials)

2. zipped **ASC II** format (*.txt.zip). Each of those files contains a 2-D matrix where each row (line) contains the data of one trial, beginning with all samples of the first channel. Channels are in the following order: (F3, F1, Fz, F2, F4, FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6, O1, O2). In the files containing training data the first entry in each row indicates the class (0: left, 1: right). In the 1000 Hz version trials consist of 500 samples per channel and in the 100 Hz version they consist of 50 samples.

## Requirements and Evaluation

Please provide your estimated class labels (0 or 1) for every trial of the test data and give a description of the used algorithm. The performance measure is the classification accuracy (correct classified trials divided by the total number of test trials).

## Technical data

The recording was made using a NeuroScan amplifier and a Ag/AgCl electrode cap from ECI. 28 EEG channels were measured at positions of the international 10/20-system (F, FC, C, and CP rows and O1, O2). Signals were recorded at 1000 Hz with a band-pass filter between 0.05 and 200 Hz.

## References

1. Benjamin Blankertz, Gabriel Curio and Klaus-Robert Müller, Classifying Single Trial EEG: Towards Brain Computer Interfacing, In: T. G. Diettrich and S. Becker and Z. Ghahramani (eds.), Advances in Neural Inf. Proc. Systems 14 (NIPS 01), 2002.

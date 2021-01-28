# Generating Music with LSTM


1.0 Introduction and Motivation
There’s a lot of incredible progress being made in the field of data science and in deep learning in particular. We are seeing the rise of different kinds of generative models for natural language processing, image creation, and much more. In general, there has been a great emphasis on how deep learning can generate text, but not as much work has been done surrounding music generation.  
In this report, we will seek to address this gap. We hope to showcase how music can not only be generated using long short-term memory (LSTM) networks, but also how certain musical styles can be emulated given the appropriate corpus of training data. Furthermore, we will compare the performance of LSTM models to other artificial neural network (ANN) architectures in order to emphasize their utility in this context. 
2.0 Background
To implement a model to generate music using an LSTM network, it is essential to have some background information on recurrent neural networks, the category of ANNs which LSTM falls into. Furthermore, we will explain a python library called Music21 which was essential for the implementation of the models discussed in this report. Lastly, some background will be given on LSTM architecture and the training algorithm used in LSTM networks: backpropagation through time. A brief summary of each is given below. 
2.1 Recurrent Neural Networks (RNN)
RNNs are a class of artificial neural network that are particularly useful at modelling sequential information. The defining feature of an RNN is that the input consists of the output from a previous time step in addition to the input of the current time step. As a result, output of a current time step is affected by what came before it, whereas in traditional neural networks the outputs are independent of the previous computations. Another feature of RNNs is that they perform the same function for every single element of a sequence. 
2.2 Backpropagation Through Time (BPTT) 
Backpropagation through time (BPTT), is an algorithm which is used to update weights in recurrent neural networks including LSTMs. BPTT works by “unrolling” all input timesteps (copies of the network which include hidden states of that timestep). This can be spatially understood as each input timestep acting as a layer. Errors are then calculated using gradient descent and are accumulated for each timestep. The network is then rolled back up and the weights are appropriately adjusted. 


2.3 Music21 Python Library 
For the purposes of this report we used a python library called Music 21. Music 21 is a toolkit that is generally used for teaching the fundamentals of music theory, generating and studying music. The library allows us to convert MIDI files from their original format to lists of notes and chords. We are also able to convert our vector outputs into MIDI format by creating Note and Chord objects in Music21.
Through Music21, we plan to extract the contents of our MIDI dataset and turn them into vectors for training purposes. We can also use it to transform the output of the neural network into musical notation which can then be further transformed into MIDI files to be played for the class.
2.4 Long short-term Memory (LSTM)

LSTMs were created as a solution to the vanishing/exploding gradient problem found in simple recurrent networks (SRN). LSTMs are able to avoid compounding errors (compounding gradients) because they do not perform the same function on each time step. By including a long term memory pathway, which can be changed and reset by current time steps, it allows for the network to change its transformation of the current input based on the inputs which came before it. This long term memory component has been shown to retain information for hundreds to thousands of timesteps back. 
Below is a diagram which shows an LSTM cell. The upper path which begins with Ct-1 is the long-term memory component and has multiple gates leading to it from the short-term memory pathway below it. These gates block or allow signal, which tells the cell when to read, write and erase the long term memory. They are analogues to logic gates in microshipcs, except rather than using digital signals, it uses an analogue signal which can be differentiated. 
 
https://commons.wikimedia.org/wiki/File:The_LSTM_cell.png


3.0 Description of Method
We are using LSTM (Long Short Term Memory) network for this exercise. LSTM is a type of RNN that learns efficiently via gradient descent and they recognize and encode long-term patterns using gating mechanisms. In cases where a network has to remember information for a long period of time (like music and text) LSTM is extremely useful. 
3.1 Data Gathering
We are using Final Fantasy, Beethoven and Jazz datasets. The datasets are in the form of midi files, which are manipulated using Music21. For music generation, the neural network has to predict which note is the next given a sequence of previous notes. This boils the problem down to a classification problem. 
3.2 Data Preparation
We load the data as an array using get_notes() function and the pseudo code is given below:
For all midi files:
1.	Get notes using Music21
2.	Add the notes to the list
The stream object in Music21 helps in getting the list of all notes and chords in the file. Pitch of every note and chord is appended using their string notation. Once we have put all the notes and chords into a sequential list, the sequences to our input network are created.We have used prepare_sequences() function for this and the pseudo code is given below:
1.	Take seq_len items from notes list
2.	Transpose it as row of input matrix
3.	Put the next element in the output matrix
This helps us create translate the notes list to 2-D input and output matrices. The input matrix is then normalized while output matrix is one-hot-encoded.
3.3 Models
We have used 4 types of layers in our model:
1.	LSTM layer - 3 layers
2.	Dropout Layer - 3 layers
3.	Dense Layer -2 layers 
Loss is calculated using categorical cross entropy as our outputs belong to a single class and we have more than two classes to work with. RMSprop optimizer is used as an optimizer to optimize our RNN.The network is trained using 100 epochs with each batch size of 64. Model checkpoints are used to save the intermediate models and generate output using them. 
3.4 Music Generation
To generate music we use generate_notes() that we used while training, but instead of training the model again, we loaded the weights that we saved during model training. The pseudo code is given below:
1.	Randomly Pick a row (sample) from input matrix
2.	Generate Prediction
3.	Add prediction to input and use sliding window to generate next prediction
Since we have a full list of note sequences at our disposal we will pick a random index in the list as our starting point, this allows us to rerun the generation code without changing anything and get different results every time.
For generation, we submit a sequence of length 100 to the network and get a prediction for the next note. This prediction is added to the input and input sequence is shifted by one every time like a sliding window. In the prediction list, if the pattern is a chord, we have to split the string up into an array of notes. The final list is then written to midi stream to get the audio output.
 
4.0 Experiments
Since we are dealing with music and the model is in a way a Generative model, evaluation for this problem becomes very difficult, which is why we decided to keep the experiments very broad in nature. 
●	Experiment 1 assesses the impact of epochs on the quality of output music
●	Experiment 2 tries to find out if the network can actually mimic the input provided to it by taking Beethoven’s and jazz music as input and keeping everything else the same
●	Experiment 3 compares the performance of LSTM with GRU
●	Experiment 4 deals with more problem specific tweaks related to music generation like the duration of notes and their offset to see the change in output audio
4.1 Experiment 1 - Varying Epochs  
The aim of this experiment is to intuitively understand the evolution of our network. We keep the architecture similar to the one in the original project and see how the output is after every few epochs. In the original project, the number of epochs is 200 but since ETA for each epoch is between 7-8 minutes, we decided to keep the maximum number of epochs to be 100.
As expected, in the initial epochs we observe that the network does not learn much which is evident from the high loss value. When listening to music generated using these model weights, it looks like the network is just repeating notes which could be due to less understanding of chords, which are a combination of notes played at the same time.
As the number of epochs increases, the quality of output music improves which is again evident from the steady decrease in loss as shown in figure . Since the output is music, evaluation becomes very difficult. But for music generated from the final few epochs, the overall tone does seem to resemble the training tracks of Final Fantasy.
		       Fig 1. Loss for Final Fantasy dataset
 
4.2 Experiment 2 - Emulating Musical Styles 
The aim of this experiment is to test whether our network could distinguish different music styles and generate music resembling those styles. We kept the architecture the same while training two sets of audio files, one contained only jazz piano music and the other one contained only Beethoven’s piano music. 
Although we used the same program with the same functions and parameters to train these two sets of files, the time to train each epoch and the loss decrease by training each epoch were very different. The Beethoven’s piano music files took about 7 minutes to train each epoch while the jazz piano music took about 10 minutes. After we trained the two sets of files with 90 epochs, we found that the loss for classical piano music was 0.6929 while the loss for jazz piano music was 2.66. We decided to stop training more epochs and start generating music using different weights because the loss decreased by about 0.001 as we trained each epoch for both files.

In the beginning, as we increased the number of epochs, the loss decreased and the quality of music generated was getting better. As the number of epochs increased over 50, we found that the decrease of loss slowed down and the quality of music did not necessarily depend on the number of epochs. Later, we noticed that the notes and chords randomly extracted from the training set affected the quality of music. In order to find the number of epochs that generate the best quality musicI, we controlled the notes and chords each time for the experiment and generated music using the weights when we trained 50, 70 and 90 epochs for both classical and jazz music respectively. We found that the music generated after training 50 epochs was more likely to be discontinued and monotonous. For classical music, the weights generated after training 90 epochs had the highest quality because it had an organized pattern and resembled the style of Beethoven’s piano music. The classical music generated after training 70 epochs had some random notes with high or low pitches affected the harmony of the music. 
Figure for classical music generated after training 70 epochs
 
Figure for classical music generated after training 90 epochs 
 
In terms of jazz music, we found that music generated after training 70 epochs had a better quality compared to the music generated after training 90 epochs. The jazz music after training 70 epochs had short rhythmic figures, repeating patterns and small variations in each repeated melody. However, the music generated after training 90 epochs rarely had any variations and the structure failed to resemble jazz music. 
Figure for jazz music generated after training 70 epochs
 
Figure for jazz music generated after training 90 epochs 
 
After selecting the best quality music for three different genres, we found the characteristics of the output music that matched the input and this showed that the network was able to reflect the characteristics of the inputs. 
4.3 Experiment 3 - GRU vs. LSTM 
This experiment aims to implement and compare the outputs generated from the Gated Recurrent Units (GRU)  and Long Short Term Memory (LSTM) algorithm.
 
GRU also uses the gating mechanism as LSTM and the network with GRU trained faster and generated higher quality output music. The reason for this is less computation within the GRU unit when compared to LSTM and the fact that GRU renders its hidden content without any restriction.
GRU also used the same Final Fantasy dataset. It took about 6 minutes and 15 secs on average to train each epoch and also the value of the loss decreased as the number of epochs increases. The value of loss for 10th epoch was around 3. 5749 and for the 100th epoch the value of loss reduced to 0.7214. On comparing the outputs of LSTM and GRU, It is quite visible that the output of GRU was better in comparison to the output of LSTM.

     Fig 2: Loss Vs Number of Epochs for GRU 

4.4 Experiment 4 - Learning Structured Improvisation of Jazz by Adding the Offset Dimension 
The purpose of this experiment is to further explore the capability of LSTM in handling the task of learning structured improvisation. i.e. learning music structures that have improvised elements to it. In this experiment, we try to replicate two specific patterns of blues -- first, modal improvisation (i.e. improvise the pattern of repetition under the constraint of a legal set of notes). Second, unresolved cadence (i.e. improvise the assignment of a special chord). (Seguin, M.-A. n.d., Jump 2018) The motivation to replicate these two patterns is that they are easily recognizable -- it will make it easier to analyze whether we successfully replicated these patterns of structured improvisation. Also structured improvisation seems to be a tricky task that will challenge the capability of LSTM. 
Two variations are made to the LSTM model to replicate these two patterns : 1) allow the offsets (the interval between two notes) to be trained. 2) pass shorter sequence into the short-term memory of LSTM, with the hope that it will capture structures that are more local. The hypothesis that the LSTM model should be able to replicate the two blues patterns and do a better job at learning jazz. The result of this experiment is that LSTM performs worse in replicating jazz than in experiment. And it was only able to replicate the pattern of unresolved cadence . 
The implication of the result is that the pattern of modal improvisation is local. LSTM did not perform well perhaps because it is trying to learn the general structure of blues as well as the local patterns of it. One suggestion for future trials is that a better way of handling this hybration of problem is transfer learning -- pre-train some layers to extract the local patterns, and then combine it with layers that handles higher level structures of the music. 

5.0 Conclusion
Musical composition involves an understanding of the underlying structure of how notes relate to one another. 
To emulate a musical style one must be able to reconstruct the structure of that style. 
Deep learning applications for generating music benefit greatly from memory components which can retain past information and help model the structure of musical styles. 







References/Further Reading:
1.	Issa.A.(2019, May 21). Generating Original Classical Music with an Lstm Neural Network and Attention. Retrieved from  https://medium.com/@alexissa122/generating-original-classical-music-with-an-lstm-neural-network-and-attention-abf03f9ddcb4
2.	Skúli, S. (2017, December 9). How to Generate Music using a LSTM Neural Network in Keras. Retrieved from https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

3.	Sigurgeirsson, S.Classical-piano-composer. (n.d.). Retrieved from 
Skuldur - https://github.com/Skuldur/Classical-Piano-Composer/tree/master/data

4.	The MAESTRO Dataset. (2018, October 29). Retrieved from https://magenta.tensorflow.org/datasets/maestro

5.	Jazz datasets
bTd Blues MIDI archive (n.d.). Retrieved from https://www.dongrays.com/midi/archive/jazz/blues/
Bob, Jack, & Dan. (n.d.). MIDKAR.COM Blues MIDI Files (Q - Z). Retrieved from http://midkar.com/blues/blues_03.html

6.	Belousov,M. (n.d.). Retrieved from 
https://github.com/mbelousov/schumann

7.	Seguin, M.-A. (n.d.). The Ultimate No Nonsense Guide to Jazz Harmony. Retrieved from https://www.jazzguitarlessons.net/blog/the-ultimate-no-nonsense-guide-to-jazz-harmony
8.	Jump, B. (2018, June 15). The Structure and Essence of Jazz. Retrieved from https://brianjump.net/2017/07/07/the-structure-and-essence-of-jazz/

9.	A Beginners Guide to LSTMs and Recurrent Neural Networks. (n.d.). Retrieved February 10, 2020, from https://pathmind.com/wiki/lstm

10.	Belousov,M.,Phuycharoen,M., Milosevic,N. (2017, January 1). Recurrent Neural Networks Composing Music. Retrieved from 
http://inspiratron.org/blog/2017/01/01/schumann-rnn-composing-music/

11.	Brownlee, J. (2019, August 14). A Gentle Introduction to Backpropagation Through Time. Retrieved February 10, 2020, from https://machinelearningmastery.com/gentle-introduction-backpropagation-time/














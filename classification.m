% Toy example of convolutional layers considered as mobile agents (Fig. 2C,D).

% First load the MNIST data set, if it hasn't already been loaded:
if ~exist('training','var')
  load mnist
end
width = 28;   % image dimensions
height = 28;

Clist = [2 3 4 6 8 11 16 23 32 46 64];  % list of values to test for C
Rlist = [3 5 7 9 11 13 15];             % list of values to test for R
replicates = 20;                        % number of trials for each combination of C and R
F = 3;                                  % side length for each such convolutional filter
test_score_record = zeros(length(Clist),length(Rlist),replicates);

for Cloop = 1:length(Clist)
  for Rloop = 1:length(Rlist)
    C = Clist(Cloop);
    R = Rlist(Rloop);
    for reploop=1:replicates

      % For this set of parameters, first create the network from scratch:
      clear net netTrained
      net = dlnetwork;
      layer = imageInputLayer([width height 1],'Name','image');
      net = addLayers(net,layer);
      % Now create a RxR layer just for reference, as the size for the cropped image subareas:
      layer = minicropLayer('mini',[R R]);
      net = addLayers(net,layer);
      net = connectLayers(net,'image','mini');
      % Create a layer that will ultimately concatenate all the separate cropped layers we're about to create:
      layer = depthConcatenationLayer(C,'Name','catfilters');
      net = addLayers(net,layer);
      % Now create C convolution filters, each one restricted to a different subarea of the input image:
      for i=1:C
	% Choose a randomly located subarea of the input image to crop:
	locx = ceil(rand*(width-R+1));
	locy = ceil(rand*(height-R+1));
	layer = crop2dLayer([locx locy],'Name',sprintf("crop%d",i));
	net = addLayers(net,layer);
	net = connectLayers(net,'mini',sprintf("crop%d/ref",i));
	net = connectLayers(net,'image',sprintf("crop%d/in",i));
	% Create a convolution layer just for that subarea:
	layer = convolution2dLayer([F F],1,'Padding','same','Name',sprintf("conv%d",i));
	net = addLayers(net,layer);
	net = connectLayers(net,sprintf("crop%d",i),sprintf("conv%d",i));
	% tanh nonlinearity:
	layer = tanhLayer('Name',sprintf("tanhcon%d",i));
	net = addLayers(net,layer);
	net = connectLayers(net,sprintf("conv%d",i),sprintf("tanhcon%d",i));
	net = connectLayers(net,sprintf("tanhcon%d",i),sprintf("catfilters/in%d",i));
      end
      % Finally, create a fully connected output layer and apply the softmax function:
      layer = fullyConnectedLayer(size(training_labels,1),'Name','full');
      net = addLayers(net,layer);
      net = connectLayers(net,'catfilters','full');
      layer = softmaxLayer(Name='out');
      net = addLayers(net,layer);
      net = connectLayers(net,'full','out');

      % Use MATLAB's provided trainnet function:
      options = trainingOptions("sgdm", MaxEpochs=4, Verbose=false, Plots="none", Metrics="accuracy");
      netTrained = trainnet(reshape(training,[width height 1 size(training,2)]),training_labels',net,"crossentropy",options);

      % Evaluate and store the trained network's accuracy:
      test_score_record(Cloop,Rloop,reploop) = testnet(netTrained,reshape(test,[width height 1 size(test,2)]),test_labels',"accuracy");

    end
  end
end

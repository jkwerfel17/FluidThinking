% Simple example of a swarm using transient immobility for distributed
% computations, in a collective perception and navigation task (Fig. 3).

% The script is self-contained, and includes the code to generate a
% data set of arrow images and to train a network. Alternatively,
% perception_and_navigation.mat (provided in the same directory as
% this file) contains a network already trained on such a data set.

% Uncommenting the following two lines will exactly reproduce the
% results shown in Figure 3 and Video 1.
% load perception_and_navigation
% rng(1)


% I once heard Douglas Adams give a talk at MIT. During the Q&A,
% someone asked him, "Why 42?" He answered, in the voice of someone
% who has been asked that question tens of thousands of times, "It is
% impossible for language to convey how meaningless that choice was."

% That comes to mind for me here because this example program is full
% of arbitrary choices. Why this architecture for the ANN? (Because it
% had to have *some* architecture, and this was like the one I was
% already using for Figure 2.) Why this movement model for the agents?
% (Because it had to have *some* movement model, and this was like the
% one colleagues and I used in our 2022 J Roy Soc Interface paper.)
% Etc. Better choices surely exist for everything; these were
% sufficient and convenient.

% One side effect of that is: these agents aren't doing an optimal job
% on this task. Sometimes they get the direction of the arrow wrong,
% etc. The purpose of the demonstration isn't for this task to work
% flawlessly, it's just to illustrate the way the task is performed.


N = 25;                % number of robots
C = 32;                % number of convolutional filters in the ANN
R = 1;                 % radius of perception for robots ("r" in the figure caption)
D = 28;                % side length of the local area each robot estimates for the direction estimation
arena_size = 250;      % side length for arena
drawn_arrow_size = 10; % length for the arrows drawn in the arena
num_examples = 1e5;    % total number of examples used for training and validation sets together
makemovie = 1;         % for convenience: boolean for creating an output movie, or not bothering


% First check to see if there's already a network present
if ~exist('netTrained','var')
  % If not, see if there's a data set to train it
  if ~exist('trainingset','var')
    % If not, create such a data set
    wb = waitbar(0,'Building data set...');
    trainingset = zeros(D*D,round(num_examples*0.8));
    traininglabels = zeros(2,round(num_examples*0.8));
    testset = zeros(D*D,round(num_examples*0.2));  % validation set
    testlabels = zeros(2,round(num_examples*0.2));
    im = zeros(D);  % blank slate that different arrows will be drawn on
    for i=1:num_examples
      % Choose a random angle for the arrow and a random position within the slate
      ang = rand*2*pi;      % angle
      ofx = D/4*(rand-.5);  % offset x
      ofy = D/4*(rand-.5);  % offset y
      % Use Matlab's built-in insertShape to draw the arrow, onto a new slate called im2:
      im2 = mean(insertShape(im,'line',D/2*(1+0.5*[-cos(ang) -sin(ang) cos(ang) sin(ang); ...  % main stem
						   cos(ang) sin(ang) cos(ang)+cos(pi+ang-pi/4) sin(ang)+sin(pi+ang-pi/4); ...  % one side of head
						   cos(ang) sin(ang) cos(ang)+cos(pi+ang+pi/4) sin(ang)+sin(pi+ang+pi/4)]) ...  % other side of head
				       + repmat([ofx ofy ofx ofy],3,1),LineWidth=1),3);  % and offset the position
      if i<=num_examples*0.8
	trainingset(:,i) = im2(:);
	traininglabels(:,i) = [cos(ang); sin(ang)];
	% Rather than predict the angle directly, we predict its cosine and sine, so that we're working with quantities that are continuous as the angle goes around. We'll use the predicted cosine and sine to recover the angle later.
      else
	testset(:,i-num_examples*0.8) = im2(:);
	testlabels(:,i-num_examples*0.8) = [cos(ang); sin(ang)];
      end
      waitbar(i/num_examples,wb);
    end
    delete(wb)  % get rid of the wait bar once the data set is complete
  end  % now we have a data set

  % Next, create and train the network (architecture shown in Fig 3A)
  net = dlnetwork;
  layer = imageInputLayer([D D 1],'Name','image');
  net = addLayers(net,layer);
  layer = dropoutLayer(0.3,'Name','drop');
  net = addLayers(net,layer);
  net = connectLayers(net,'image','drop');
  layer = convolution2dLayer([2*R+1 2*R+1],C,'Name','conv');
  net = addLayers(net,layer);
  net = connectLayers(net,'drop','conv');
  layer = reluLayer('Name','relu');
  net = addLayers(net,layer);
  net = connectLayers(net,'conv','relu');
  layer = fullyConnectedLayer(C,'Name','extra');
  net = addLayers(net,layer);
  net = connectLayers(net,'relu','extra');
  layer = reluLayer('Name','relu2');
  net = addLayers(net,layer);
  net = connectLayers(net,'extra','relu2');
  layer = fullyConnectedLayer(2,'Name','out');
  net = addLayers(net,layer);
  net = connectLayers(net,'relu2','out');

  options = trainingOptions("sgdm", ...
			    MaxEpochs=15, ...
			    Verbose=true, ...
			    Plots="training-progress", ...
			    ValidationData={reshape(testset,[D D 1 size(testset,2)]),testlabels'}, ...
			    Metrics="rmse");

  netTrained = trainnet(reshape(trainingset,[D D 1 size(trainingset,2)]),traininglabels',net,"mae",options);
  accuracy=testnet(netTrained,reshape(testset,[D D 1 size(testset,2)]),testlabels',"mae");
end


% At this point we have a trained network for recognizing arrow direction. Proceed:

if makemovie  % set up the output movie if we're making one
  mov = VideoWriter('Figure3.mp4','MPEG-4');
  open(mov)
end
    
% Create the arena, and draw a series of arrows in it with given angles and positions
arena = zeros(arena_size);
angs = [130 -25 -130 100 -10 110 185 -115 0]*2*pi/360;
ofxs = [.45 .25 .85 .55 .55 .9 .85 .2 .1]*arena_size;
ofys = [.2 .55 .5 .15 .7 .68 .9 .85 .6]*arena_size;
for i=1:length(angs)
  ang = angs(i); ofx = ofxs(i); ofy = ofys(i);
  arena = mean(insertShape(arena,'line',D/2*(0.5*[-cos(ang) -sin(ang) cos(ang) sin(ang); cos(ang) sin(ang) cos(ang)+cos(pi+ang-pi/4) sin(ang)+sin(pi+ang-pi/4); cos(ang) sin(ang) cos(ang)+cos(pi+ang+pi/4) sin(ang)+sin(pi+ang+pi/4)]) + repmat([ofx ofy ofx ofy],3,1),LineWidth=1),3);  % this is the same way we drew it when creating the data set, above
end
initx = ofxs(1); inity = ofys(1);  % initial location of first arrow, to put swarm near at start
% Initialize the robots, located near first arrow, and pointing toward it (with some variation)
clear a
for i=1:N
  a(i).x = initx - 3*D + rand*N;
  a(i).y = inity + (rand-.5)*N;
  a(i).ang = (rand-.5)*.25;
  a(i).color = rand(1,3);  % for visualization, to help discriminate between robots
  if makemovie, a(i).color = [1 0 0]; end  % but a movie is arguably cleaner with an identical swarm
  % For convenience, we'll draw robots using Erik Johnson's arrow.m (https://www.mathworks.com/matlabcentral/fileexchange/278-arrow)
  a(i).ar = arrow([0 0],[1 1]);
  a(i).offset = rand*50;  % this will be used later to add some randomness to the robots' movement, persistent but unique to each robot
  a(i).targang = (rand-.5)*.05;  % target angle for each robot: horizontally to the right (with a little variation)
end

% Draw the arena
imagesc(-arena)
colormap(gray)
axis equal
axis off
set(gcf,'color',[1 1 1])

% Repeat enough times to try to follow each arrow in the list set above
for bigloop=1:length(angs)+1

  % This first stage applies before any robot encounters the arrow
  % -> There are a few sub-parts of this stage, tracked using a mode variable called flag.
% Because (other than right at the start) robots are executing this code immediately after having decided on their next direction based on the last arrow, we want a refractory period to keep them from immediately zeroing in on the same arrow again. flag==-1 indicates being in that refractory period. flag==0 indicates that period is over, and robots are looking for the next arrow marked in the arena. flag==1 indicates that some robot has found it. Then we wait a few extra time steps to let the swarm move on a little, so others behind the leading robot can also get closer to the arrow; after that, flag gets set to 2 and robots move on to the next stage.
  flag = -1;
  t = 2*D;   % variable for keeping track of time within this stage; here we set it to that refractory period
  lead = 0;  % this will store the index of the robot that first saw the arrow
  while flag<2
    for i=randperm(N)  % go through the robots in random order
      % look at the region within this robot's sensing range: (note that x and y are reversed here, as an artifact of mostly treating the arena as an image (x,y) but here treating it as a matrix (row,col))
      view = arena(max(1,round(a(i).y-R)):min(arena_size,round(a(i).y+R)),max(1,round(a(i).x-R)):min(arena_size,round(a(i).x+R)));
      if i==1 && flag==-1  % wait a few time steps before actually starting to look for arrows, per comment above
	t = t-1;
	if t==0
	  flag=0;
	end
      end
      if flag==0 && ~isempty(find(view(:)))  % when some robot sees something, count down a few more time steps...
	flag = 1;
	lead = i;
	t = 3;
      end
      if i==lead && flag==1  % ...and then move on to the next stage
	t = t-1;
	if t<=0
	  flag = 2;
	end
      end

      % The position/angle update will be a variation on a classic swarm movement model, as implemented in Joshi et al. 2022: robots move straight ahead, and turn toward their target angle, but also turn away from neighbors they're too close to (rudimentary collision avoidance)
      fxn = 0.9*cos(a(i).targang); fyn = 0.9*sin(a(i).targang);  % initialize new desired angle as vector in goal direction
      lst = find(sum((repmat([a(i).x; a(i).y],1,N)-[[a.x]; [a.y]]).^2) < 5*R*R);  % IDs of nearby robots
      ind = find(lst==i); lst = [lst([1:ind-1 ind+1:end])];  % remove self from that list
      for j=1:length(lst)  % for each neighbor,
	nx = a(lst(j)).x - a(i).x; ny = a(lst(j)).y - a(i).y;  % take the vector to that neighbor,
	nxn = nx / sqrt(nx^2+ny^2); nyn = ny / sqrt(nx^2+ny^2);  % normalize it,
	fxn = fxn - nxn; fyn = fyn - nyn;  % and add to the accumulator a vector in the opposite direction
      end
      ang = atan2(fyn,fxn);  % new desired angle based on all influences (goal + neighbors)
      delangdes = angdiff(ang,a(i).ang);  % turn toward that desired angle, left or right according to which is closer
      a(i).ang = a(i).ang - min(abs(delangdes),0.25)*sign(delangdes);  % robots have a maximum turning rate
      speed = (1 - delangdes/pi);  % have robots go faster when they're heading straight, and slow down when they're turning hard
      a(i).x = a(i).x + speed*cos(a(i).ang);  % finally, update position
      a(i).y = a(i).y + speed*sin(a(i).ang);
      delete(a(i).ar);  % update where the robot is drawn
      a(i).ar = arrow([a(i).x-.1*cos(a(i).ang) a(i).y-.1*sin(a(i).ang)],[a(i).x+.1*cos(a(i).ang) a(i).y+.1*sin(a(i).ang)],'facecolor',a(i).color,'length',10);
    end  % looping through all robots
    drawnow
    if makemovie, frame = getframe(gcf); writeVideo(mov,frame), end
  end

  % Done with the first stage: the robots have found an arrow. Next, they gather around it, to try to come up with composite pictures of the region.
  for i=1:N
    a(i).moving = 1;  % once a robot thinks it's in a good location, it will stop moving
    a(i).targx = a(lead).x;  % now it heads for a target location, initially that of the robot that first saw the arrow
    a(i).targy = a(lead).y;
  end
  a(lead).moving = 0;  % that robot that first saw the arrow stops right away
  t = 0;  % variable for keeping track of time within this stage
  while ~isempty(find([a.moving]))  % repeat until everyone has stopped
    t = t+1;
    for i=randperm(N)
      if a(i).moving
	% Position/angle update -- essentially the same as above, with a target position rather than target angle
          fx = a(i).targx - a(i).x; fy = a(i).targy - a(i).y;  % initialize target vector toward goal
	  fxn = fx / sqrt(fx^2+fy^2); fyn = fy / sqrt(fx^2+fy^2);  % normalized
	  dist = [[a.x]; [a.y]];
	  lst = find(sum((repmat([a(i).x; a(i).y],1,N)-dist).^2) < 5*R*R);  % IDs of nearby robots
	  ind = find(lst==i); lst = [lst([1:ind-1 ind+1:end])];  % remove self from that list
	  for j=1:length(lst)
	    nx = a(lst(j)).x - a(i).x; ny = a(lst(j)).y - a(i).y;  % take the vector to that neighbor,
	    nxn = nx / sqrt(nx^2+ny^2); nyn = ny / sqrt(nx^2+ny^2);  % normalize it,
	    fxn = fxn - nxn; fyn = fyn - nyn;  % and add to the accumulator a vector in the opposite direction
	  end
	  ang = atan2(fyn,fxn);  % new desired angle based on all influences (target position + neighbors)
	  delangdes = angdiff(ang,a(i).ang);  % turn toward that desired angle, again with a maximum turning rate
	  a(i).ang = a(i).ang - min(abs(delangdes),0.25)*sign(delangdes) + (.05+.2*(sin((t-a(i).offset)/50)+1))*(rand-.5);
	  speed = (1 - delangdes/pi);  % same modulation of speed based on turning
	  a(i).x = a(i).x + speed*cos(a(i).ang);  % update position
	  a(i).y = a(i).y + speed*sin(a(i).ang);

	  % The robot looks at the locally visible area of the arena as it moves:
	  viewi = arena(max(1,round(a(i).y-R)):min(arena_size,round(a(i).y+R)),max(1,round(a(i).x-R)):min(arena_size,round(a(i).x+R)));
	  if ~isempty(find(viewi(:)))  % ...and if it sees part of the arrow, then maybe this location can use coverage:
	    a(i).targx = 0.7*a(i).targx + 0.3*a(i).x;  % update the goal location to move in the direction of what it saw
	    a(i).targy = 0.7*a(i).targy + 0.3*a(i).y;
	  end

	  % Robots stop if they're close enough to their goal, another robot is stopped in front of them, and no others are already stopped too close
	  dist = [[a.x]; [a.y]];
	  lst = find(sum((repmat([a(i).x; a(i).y],1,N)-dist).^2) < 15*R*R);  % IDs of medium-nearby robots
	  ind = find(lst==i); lst = [lst([1:ind-1 ind+1:end])];  % remove self from that list
	  lst2 = find(sum((repmat([a(i).x; a(i).y],1,N)-dist).^2) < 4*R*R);  % IDs of nearby robots
	  if t<=5*N  % using the above conditions and this movement model, sometimes a robot can get trapped circling in the middle of a crowd, so we also include a timeout
	    if isempty(find([a(lst2).moving]==0))  % if no one is stopped too close, look for neighbors in vision cone
	      for j=1:length(lst)
		angj = atan2(a(lst(j)).y-a(i).y,a(lst(j)).x-a(i).x);  % direction to this neighbor
		dist = (a(lst(j)).x-a(i).x)^2+(a(lst(j)).y-a(i).y)^2;  % distance to this neighbor
		dfg = (a(i).x-a(i).targx)^2+(a(i).y-a(i).targy)^2;  % distance from individual's goal
		if abs(angdiff(a(i).ang,angj)) < pi/6 && (a(lst(j)).moving==0 && dist > 4*R*R) && dfg < sqrt(7*N/pi)*R
		  a(i).moving = 0;
		end
	      end
	    end
	  else a(i).moving = 0;  % timeout
	  end

	  delete(a(i).ar);
	  a(i).ar = arrow([a(i).x-.1*cos(a(i).ang) a(i).y-.1*sin(a(i).ang)],[a(i).x+.1*cos(a(i).ang) a(i).y+.1*sin(a(i).ang)],'facecolor',a(i).color,'length',10);
      end  % if robot i is moving
    end  % loop over all robots
    drawnow
    if makemovie, frame = getframe(gcf); writeVideo(mov,frame), end
  end  % repeat until everyone has stopped moving


  % Done with the second stage: the robots have taken up fixed positions around the arrow. Next, they each build a composite map of the area, and use the pretrained network to recognize the direction of the arrow.
  % (Strictly speaking, we would want each robot to build up its own map relative to its own position and angle, based on the sensed distances and angles between robots, as with Moore et al. (SenSys 2004) or Rubenstein et al. (Science 2014). However, for purposes of this demo, we'll abstract that away and just say that robots construct a noisy estimate of others' locations, with noise increasing with distance.)
  for i=1:N
    arena3 = zeros(arena_size);  % this will become the robot's map of the area (making it this size rather than DxD for convenience)
    for j=1:N
      dist = sqrt((a(j).x-a(i).x)^2+(a(j).y-a(i).y)^2);   % actual distance to neighbor
      neix = (a(j).x-a(i).x) * (1+(rand-.5)*dist/D*0.5);  % estimate their location, with noise based on distance
      neiy = (a(j).y-a(i).y) * (1+(rand-.5)*dist/D);      % (choosing a relatively high noise scale, such that if a neighbor is D away, the estimated offset can be off by 50%)
      neix = round(neix); neiy = round(neiy);
      % Now that we know where (we think) they are, put their (actually-located) map into ours:
      arena3(arena_size/2+neiy-R:arena_size/2+neiy+R,arena_size/2+neix-R:arena_size/2+neix+R) = arena(round(a(j).y)-R:round(a(j).y)+R,round(a(j).x)-R:round(a(j).x)+R);
    end
    view = arena3(arena_size/2-D/2:arena_size/2+D/2-1, arena_size/2-D/2:arena_size/2+D/2-1);  % the DxD area around the robot
    % Having pieced together an estimate of the area by putting what all robots actually perceive where we think they are, see what direction the pretrained network thinks that arrow is:
    p = predict(netTrained,view);
    angpred = atan2(p(2),p(1));
    a(i).targang = angpred;
  end

  % Now all N robots have an estimated angle of the arrow; each chooses a new target direction to move next as a weighted combination of those estimates:
  xlist = cos([a.targang]);
  ylist = sin([a.targang]);
  average_ang = atan2(sum(ylist),sum(xlist));
  for i=1:N
    a(i).targang = atan2(sin(average_ang)+0.33*sin(a(i).targang),cos(average_ang)+0.33*cos(a(i).targang));
  end

end  % loop over all arrows in the list

/* % Toy example of mobile agents performing a consensus task (Fig. 2A,B). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define d 5                // size of arena
#define P 0.6              // fraction of squares of one color; should be >= 0.5
#define replicates 100     // number of trials
#define dt 1.0             // time step
#define tmax 32400         // upper limit on how long to run for (32400 seconds = 9 hours)
#define commrad 0.3        // communication range (radius)
#define meanlen 500        // period over which to look at running means to check for early termination (see below)
#define N 1000             // number of agents
#define v 0.1              // speed of agents

// macros for random numbers, uniform and Gaussian:
#define rnd ((double)random()/((double)RAND_MAX+1.0))
#define rndn (sqrt(-2*log(rnd))*cos(6.28*rnd))

struct Agent {
  double x, y, dir;        // position, direction
  int turning;             // binary state for switching between moving forward and rotating
  double target;           // how far to move/rotate before switching
  double val, nextval;     // val is the estimate of the global mean color; nextval is used for synchronous update
  int nei[300];            // list of neighbors (inelegantly hard-coding a limit, for convenience)
};

int main (int argc, char **argv) {

  int i, j, k, Nloop, vloop, reploop, t, subt;           // index variables
  int arena[d][d];                                       // randomly tiled with binary values
  double w, sumw, stdmx1, stdmx2, stdmn1, stdmn2, mean;  // various cryptically named intermediate variables
  
  double consensus_success_record[replicates];
  double thisrunmax[tmax], thisrunmin[tmax], thisrunmean[tmax], fraccorrect[tmax];  // keep track of various values at each time step of the run
  struct Agent a[N];

  if (argc>1) srand(atoi(argv[1]));  // allow a random seed to be specified at the command line (optional)
  
  for (reploop=0; reploop<replicates; reploop++) {
      
      // Initialize the arena, setting a fraction P of its tiles to 1 and the rest to 0:
      for (i=0; i<d; i++)
	for (j=0; j<d; j++)
	  arena[i][j]=0;
      for (k=0; k<round(P*d*d); k++) {  // choose a random location not yet chosen, P*d*d times
	i = (int)(rnd*d);
	j = (int)(rnd*d);
	while (arena[i][j]==1) {
	  i = (int)(rnd*d);
	  j = (int)(rnd*d);
	}
	arena[i][j] = 1;
      }
      
      // Initialize the agents, with random location, direction, moving/turning state, and distance:
      for (i=0; i<N; i++) {
	a[i].x = rnd*d;
	a[i].y = rnd*d;
	a[i].dir = rnd*2*3.14159;
	a[i].turning = round(rnd);
	a[i].target = a[i].dir + rnd*2*3.14159 - 3.14159;
	a[i].val = arena[(int)(a[i].x)][(int)(a[i].y)];   // initial estimate of the mean arena color is the color at the agent's position
	a[i].nei[0] = 0;                                  // entry 0 of the neighbor list specifies the current number of neighbors
      }
      
      for (t=0; t<tmax; t++) {

	/* Rather than always running for tmax steps, we may break out early if the population has converged.
	   To evaluate that (see full criteria below), we'll keep track of the mean val in the population
	   at each time step, and the spread in the population above and below that mean.
	   Also, keep track of the fraction of the population that has a correct estimate at each time step. */
	thisrunmax[t] = 0; thisrunmin[t] = 1.0; thisrunmean[t] = 0;
	j = 0;  // using as a temporary counter for number of agents with correct estimate
	for (i=0; i<N; i++) {
	  if (a[i].val > thisrunmax[t]) thisrunmax[t] = a[i].val;
	  if (a[i].val < thisrunmin[t]) thisrunmin[t] = a[i].val;
	  thisrunmean[t] += a[i].val;
	  if (a[i].val>0.5) j++;
	}
	thisrunmean[t] /= N;
	fraccorrect[t] = ((double)j)/N;
	thisrunmax[t] -= thisrunmean[t];
	thisrunmin[t] -= thisrunmean[t];
		
	// Now loop through all agents:
	for (i=0; i<N; i++) {
	  
	  // First, update pose (movement/turning speeds/distances, communication range, etc. are chosen following ref. [94]):
	  if (a[i].turning)                    // if turning in place, keep turning until the total angle turned reaches the target
	    if (a[i].dir < a[i].target) {
	      a[i].dir += 0.63*dt;
	      if (a[i].dir > a[i].target) {    // once the target is reached, switch to moving, and set a target distance
		a[i].turning = 0;
		a[i].target = 240.0 + 10.0*rndn;
	      }
	    }
	    else {
	      a[i].dir -= 0.63*dt;
	      if (a[i].dir < a[i].target) {
		a[i].turning = 0;
		a[i].target = 240.0 + 10.0*rndn;
	      }
	    }
	  else {                               // if moving forward, keep moving until the total distance traveled reaches the target
	    a[i].x += v*cos(a[i].dir)*dt/6.0;
	    a[i].y += v*sin(a[i].dir)*dt/6.0;
	    if (a[i].x > d) {                  // reflect off arena walls
	      a[i].x = d - (a[i].x-d);
	      a[i].dir = 3.14159 - fmod(a[i].dir, 6.283);
	    }
	    if (a[i].x < 0) {
	      a[i].x = -a[i].x;
	      a[i].dir = 3.14159 - fmod(a[i].dir, 6.283);
	    }
	    if (a[i].y > d) {
	      a[i].y = d - (a[i].y - d);
	      a[i].dir = -a[i].dir;
	    }
	    if (a[i].y < 0) {
	      a[i].y = -a[i].y;
	      a[i].dir = -a[i].dir;
	    }
	    a[i].target -= dt/6.0;
	    if (a[i].target < 0) {             // once the target is reached, switch to turning, and set a target angle
	      a[i].turning = 1;
	      a[i].target = a[i].dir + rnd*6.283 - 3.14159;
	    }
	  }
	}

	// After everyone moves, determine a new list of current neighbors:
	for (i=0; i<N; i++)
	  a[i].nei[0] = 0;                 // initialize the neighbor list to be empty
	for (i=0; i<N; i++) {
	  for (j=i+1; j<N; j++)
	    if (pow(a[i].x-a[j].x,2) + pow(a[i].y-a[j].y,2) < pow(commrad,2)) {
	      a[i].nei[0]++;               // if two agents are within the communication range, put them in each other's neighbor lists
	      a[i].nei[a[i].nei[0]] = j;
	      a[j].nei[0]++;
	      a[j].nei[a[j].nei[0]] = i;
	    }
	}

	// Update all values based on communication with neighbors, using the equation given in the text:
	for (i=0; i<N; i++) {
	  a[i].nextval = 0;
	  sumw = 0;
	  for (j=1; j<=a[i].nei[0]; j++) {
	    w = 1/(fmax(a[i].nei[0],a[a[i].nei[j]].nei[0])+1.0);
	    sumw += w;
	    a[i].nextval += w * a[a[i].nei[j]].val;
	  }
	  a[i].nextval = 0.001*arena[(int)(a[i].x)][(int)(a[i].y)] + 0.999*(a[i].nextval + (1.0-sumw)*a[i].val);
	}

	for (i=0; i<N; i++) a[i].val = a[i].nextval;   // synchronous update: all agents update with the newly calculated val

	/* Check to see if the dynamics of val in the population appear to have reached a steady state.
	   If so, we can stop early. The criterion is: look back over the last meanlen time steps,
	   calculate the standard deviation of both (max - mean) and (min - mean) in the population over that period,
	   and do the same over the previous meanlen time steps before that.
	   If the earlier standard deviation is less than 1.2 times the later standard deviation for both quantities,
	   the dynamics can be considered to have converged, at the start of the earlier period, i.e., 2*meanlen steps ago.
	   (1.2 is a somewhat arbitrary value, chosen empirically based on visual inspection of many runs.)
	   This criterion works well in practice for v>0 runs, for which there is ongoing variability in the population
	   values as agents move around and neighbor relationships change. For v=0 runs, all agents in the population
	   approach a steady-state value asymptotically; a separate criterion for this case is if the standard deviation
	   of (max - mean) over the earlier period is less than 1e-5 (another value chosen empirically based on visual
	   inspection of many runs). Empirically, these criteria identify that a run has reached a steady state at a time
	   which is a good match for when a human observer would consider it to have converged. */
	if (t>2*meanlen) {
	  mean = 0;
	  for (subt=t-2*meanlen; subt<=t-meanlen-1; subt++)
	    mean += thisrunmax[subt];
	  mean /= meanlen; stdmx1 = 0;
	  for (subt=t-2*meanlen; subt<=t-meanlen-1; subt++)
	    stdmx1 += pow(thisrunmax[subt]-mean,2);
	  stdmx1 = sqrt(stdmx1/meanlen);
	  mean = 0; 
	  for (subt=t-meanlen; subt<=t; subt++)
	    mean += thisrunmax[subt];
	  mean /= meanlen; stdmx2 = 0;
	  for (subt=t-meanlen; subt<=t; subt++)
	    stdmx2 += pow(thisrunmax[subt]-mean,2);
	  stdmx2 = sqrt(stdmx2/meanlen);
	  mean = 0;
	  for (subt=t-2*meanlen; subt<=t-meanlen-1; subt++)
	    mean += thisrunmin[subt];
	  mean /= meanlen; stdmn1 = 0;
	  for (subt=t-2*meanlen; subt<=t-meanlen-1; subt++)
	    stdmn1 += pow(thisrunmin[subt]-mean,2);
	  stdmn1 = sqrt(stdmn1/meanlen);
	  mean = 0; 
	  for (subt=t-meanlen; subt<=t; subt++)
	    mean += thisrunmin[subt];
	  mean /= meanlen; stdmn2 = 0;
	  for (subt=t-meanlen; subt<=t; subt++)
	    stdmn2 += pow(thisrunmin[subt]-mean,2);
	  stdmn2 = sqrt(stdmn2/meanlen);
	  if ((stdmx1 < 1.2*stdmx2 && stdmn1 < 1.2*stdmn2) || stdmn1<1.0E-5){
	    break;
	  }
	}
	
      }  // loop over time
      
      /* We've established that the population reached steady state at time (t-2*meanlen).
	 Print out the fraction of the population that had the correct value at that time: */
      printf("%g\n", fraccorrect[(int)(t-2*meanlen)]);
      fflush(stdout);
      
  }  // loop over replicates
  
  return 0;
}

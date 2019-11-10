# fiedler-sparse-cut
By Robertas Dereskevicius 2019/11
The following efficient software takes an RGB image as an input, resizes it, blurs it, 
computes the second smallest eigenvector and finds a sparse cut.
Additionally a file has been attached which contains proof for Part A.

Software has the following dependencies:
scipy, numpy, OpenCV, sklearn, matplotlib.
It was written and must be run using Python 3!
Please, make sure to run it using Python 3 as Python 2 libraries produce results which
are significantly worse and inaccurate.

HOW TO RUN THE SOFTWARE:
1) Run the following command: python3 findsparse.py
2) Input name of the image file that is in the same folder as the script (relative path works too)
	The script expects a RGB image as an input.
3) Wait for under 10 seconds and view the three freshly computed images of the blurred image, Fiedler vector
	and sparse cut. Read the console for additional output data and elapsed times!
	
Important notes and observations:
The software has been optimized to run under 10 seconds, however, the parameters are far from optimal.
The radius between pixels is set to 5, however, setting it to 7 produces significantly better results
for smaller cuts and slightly better ones for large. However, it can increase the performance time 
to over 10 seconds and can't be used at the moment. Additionally, 
Gaussian blur has been modified to only have a kernel of 3x3 due to smaller targets such as sheep or shuttle.
A larger kernel such as 5x5 would be significantly better for images like bear as it would become blurier
and have significantly less random noise.
k is set to 1500 in order to save time. In reality, it would be more accurate if the
k value was increased further, however, doubling it increases the run time for eigenvector computations
twofold (1500 is 4 seconds, 3000 is 8 seconds).
An important note to say is that occasionally given a really unlucky scenario (due to the random nature
of the algorithm) the script may run for over 10 seconds, however, it is programmed to perform for 
around 9 seconds on average.
The algorithm produces a very accurate result for on average 9 out of 10 times, however, the 10th time
can produce a very poor evaluation of the Fiedler vector. The reason for this is unclear, however, I 
believe that is caused by insufficient k and pixel radius due to time constraints in unlucky vector
initialization cases. In that case I recommend running the algorithm again to produce a good result.

Final thoughts:
I believe this algorithm runs incredibly fast and has a very good sparse cut estimation.
It has some minor bugs that could be fixed in the future and parameters that could further
improve the result with longer time limitations.

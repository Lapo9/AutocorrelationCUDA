# AutocorrelationCUDA
**Student:** Lapo Falcone
**Supervisor:** Gianpaolo Cugola

## Objectives
_The goal of the project was to write a software algorithm to calculate the autocorrelation function extremely efficiently._

### Background
Fluorescence correlation spectroscopy (FSC) is a technique used by scientists to study the behavior of biomolecules. The main advantage of this kind of analysis is to be able to monitor the experiment in real-time. The main disadvantage is the same. Indeed, in order to collect and process data as the experiment goes on, using highly specialized hardware components is mandatory. This significantly raises the costs to conduct the experiments, denying to many non-commercial scientific hubs the possibility to contribute to scientific progress, and to form new professionals.

The progress in commercial hardware that took place during the last decade, led many software developers and scientists to start thinking about a different solution: implement the well known compute-intense algorithms, which used to run on dedicated hardware, as software. This not only allowed a faster distribution of the new iterations of the tools involved in scientific analysis, but also did cut down the costs to perform the experiments. First and foremost, a commercial piece of hardware is way cheaper than a custom ASIC or FPGA. Second, a CPU or GPU is less specialized than an integrated circuit, and it can be used for a wide range of studies.

In particular, GPUs played the most important role in the regard of the transition from high-cost highly specialized hardware to low-cost generic software. Indeed GPUs are highly parallel pieces of hardware, so it is possible to analyze huge amount of data concurrently, ensuring high throughput, which is key in several scientific experiments, such as FSC.

On the other hand GPU programming is not straight forward as CPU programming, and, until recent, it wasn't open to everyone. The main pitfall of GPU programming lies in the same concept that makes GPUs suitable for scientific purposes: parallelism. The great number of cores a GPU exposes, makes most of the common programming paradigms amiss, and forces the developer to think differently. Memory layout and management becomes a key aspect, and the flow of the code must be optimized to avoid divergent execution (namely, avoid if(s)). These aspects will be analyzed soon.

### The experiment
The set-up for a FSC experiment includes these components:
* A matrix of photosensors. In our case we assumed a square matrix of 2^n photosensors.
* A server, where to analyze data provided by the matrix.
* A client, where to show the processed data.

Our interest resides on the server, so on the data processing.

The mathematical tool used to process data is the autocorrelation function, in particular in its discrete form. Simply put, the autocorrelation is a function that correlates a given input to itself. In a more formal way, given a signal `y(t)`, autocorrelation can be defined as: `A(lag) = summation[n in N](y(n) * y(n-lag))`

So a big value at a specific lag (let's call it `L`) means that the signal `y(t)` tends to be periodic, and has `L` as period. For example, the autocorrelation of `sin(x)` would yield a maximum at `2pi` and its multiples. The autocorrelation tells us how much "periodic" a function is at a given "period". The periodicity of the data collected by the matrix provides valuable information to the scientists who run the experiment.

#### Target
Thanks to the information given us by Ivan, the responsible of the FSC experiment at Politecnico di Milano, we made this assumptions:
* The matrix is made up of 32x32 photosensors, for a total of 1024 photosensors to process in parallel.
* The frequency of each photosensor is 80MHz, so the period is 12.5 nanoseconds.
* Each photosensor of the matrix send us how many photons are detected in 128 periods (1600 nanoseconds). So each value fits in 1 byte, since it holds a number between 0 and 128.
* For scientific purposes it is needed to calculate the autocorrelation up to lag 10000.


#### First approach
The first approach to write the algorithm was the trivial one: we tried to calculate `y(n) * y(n-lag)` for each instant in time for each sensor. The pro of this approach is that the code is really simple and elegant, the con is that GPUs are fast, but not this fast. Indeed, despite the fact that we asked the GPU to perform a series of simple additions and multiplications, the throughput of the GPU wasn't high enough, and we ended up processing data 3000 times slower than our target. In numbers, we set-up an environment to calculate the autocorrelation up to lag 10000 of a signal made up of 625000 values. We expected the program to take 1 second to execute. It took 3 seconds, and it analyzed only data from 1 sensor. `3 * 1024` times slower that our goal.

At this point it was clear that the approach was wrong, so we decided to look for a different way to calculate the autocorrelation of a discrete signal.

#### Multi-tau approach
After some research, we found out that it was possible to calculate an approximative form of the autocorrelation using a multi-tau approach. The fundamental drawback of this technique is that it calculates the by-definition autocorrelation only for lag values smaller than an arbitrary threshold. Then it calculates an approximate form of the autocorrelation, and the higher the lag value is, the less precise the corresponding autocorrelation will be.

The choice of the threshold is critical, since a small threshold guarantees an high throughput, but an higher threshold ensures a more accurate output. Ideally, choosing a threshold tending to infinity leads to an algorithm equivalent to the naive one.

Going a bit more into the details, as we mentioned in the matrix specifications, data is sent to the server as 8 bits integers, each containing the number of photons detected in a 1600ns period. The algorithm calculates the output as `y(n) * y(n-lag)` as long as `lag` is less than the chosen threshold (`Th`). For lag values higher than `Th` and smaller than `Th + 2Th`, the algorithm coalesce 2 periods into a single one, namely creating a single 3200ns period, which contains the sum of the values contained in the original periods. While autocorrelating lag values in the interval `Th < lag < 3Th`, the program uses these longer periods (also called bins), consequently being able to calculate the autocorrelation for lag values in a `3Th - Th = 2Th` interval in the same time it took to compute the by-definition autocorrelation for the first `Th` lag values, doubling the throughput. The drawback resides in the resolution of the output. Then the process repeats for lag values in the interval `3Th < lag < 3Th + 4Th`, by coalescing, each time, 2 3200ns bins in a single 6400ns bin, and so on.

**Lag interval** | **Bin resolution** | **Coalesced bins**
---|---|---
0 ~ Th | 1600ns | 1
Th ~ 3Th | 3200ns | 2
3Th ~ 7Th | 6400ns | 4
7Th ~ 15Th | 12800ns | 8

##### Multi-tau algorithm
In order to implement the formal algorithm in an efficient way, we had to create an ad-hoc data structure.

###### Bin group multi sensor memory
This data structure has the responsibility to hold data while being processed, and to maintain this data following the multi-tau rules. So it is its responsibility to coalesce the bins based on the threshold.

ADD IMAGE

There is such structure for each sensor of the matrix.

Particular attention should be paid to the zero delay registers. Since data in a bin group different from the 1st is coalesced, as explained in the previous paragraph, also the value corresponding to the current point in time should be coalesced. In a more formal way, for example, in the 2nd bin group we have values made up of the sum of 2 original values: `y(n-lag) + y(n-lag-1)`. In order to have the same "distance" in time between the current point in time and these coalesced values, we must also coalesce the current instant in time. So the autocorrelation for the second group becomes: `(y(n) + y(n-1)) * (y(n-lag) * y(n-lag-1))`, where the first addendum is hold by the zero delay register of the 2nd bin group. It is clear that the first zero delay register simply holds the last value added to the structure.

Moreover the first cell of each group is called accumulator, and it is the place where 2 values from the previous bin group are summed together to obtain the correct bin size for a specific group.

The interface of the structure is the following:
* `insertNew(sensor, value)`: Inserts the new value in the first position of the first bin group and in the first zero delay register. Moreover it adds the new value to the zero delay registers of all the remaining bin groups.
* `shift(sensor, binGroup)`: Shifts all of the values in the specified bin group one position to the right. The value previously present in the rightmost position leaves the group, and is added to the accumulator of the next group. Accumulator and zero delay register of the group are cleared.
* `get(sensor, group, position)`
* `getZeroDelay(sensor, group)`

In order to improve performance, 2 optimization routes were taken:
* **Reverse shift:** The shift method actually doesn't moves all of the values to the right, but actually moves the logical first position of the array to the left. In order to keep track of all of the first positions, a new array is allocated. This optimization forced another improvement: since from now on we had to work with circular arrays, we must use modulo operator, which is very expensive. Luckily it can be optimized if the right-hand-side operand is a power of 2: `x % y = x & (y-1)`.
* **Bank conflicts avoidance:** In order to use the potential of CUDA architecture fully, we arranged the arrays relative to the bin groups of each sensor "vertically", instead of "horizontally". So, after the last position of the first bin group of the first sensor, there is the first position of the first bin group of the second sensor, and not the first position of the second group of the first sensor. This way it was impossible for cells accessed concurrently to end up on the same bank.

![](https://drive.google.com/file/d/1GEaT35-h1PgZGzsNqLCxzecMywaVorrX/view?usp=sharing)


###### Algorithm
The algorithm was thought to maximize the number of operations to run concurrently. Since, in order to compute autocorrelation for bin group `x`, bin group `x-1` must be finished, we ended up to parallelize on the computation of autocorrelation of values on the same bin group. We chose the critical size of the bin group being 32, which turned out to be the right compromise between high throughput and output accuracy. Since we were bound to 1024 sensors, it means that 32768 multiplications and additions could run in parallel. This is way more than the number of CUDA cores in any of the most recent Nvidia GPUs, so this level of concurrency alone maximized the occupancy of any modern GPU.

We then chose to create 2D CUDA blocks sized 32*8. This means that on the x coordinate we have CUDA cores which work concurrently on the calculation of different lag values for the same sensor. In the y coordinate we could handle 8 sensors per block. The size of the CUDA block is then 256, which turned out to maximize the number of registers and shared memory available on each block.

Last, the choice of 32 as the size of a bin group also makes it possible to have all of the threads working on the same warp. This gains particular importance during the computation of the autocorrelation (see Computation.3.1).

* **Set-up:** Each CUDA block allocates and copy only data from sensors residing on that CUDA block.
  1. Allocate array for output and bin group multi sensor memory on shared memory.
  2. Copy output and bin group multi sensor memory passed as inputs (from previous call to this function), from global memory to shared memory.
* **Computation:** Loop that repeats for each value in the input array. Next steps happen for sure concurrently for data from each sensor residing on the same CUDA block.
  1. Insert new datum from input array to bin group multi sensor memory: `insertNew(sensor, value)`.
  2. Compute how many bin groups have to be calculated during this iteration (`T`). The first bin group is calculated on every iteration, the second one once yes and once no, the third one once every 4 iterations, and so on.
  3. `for i < T`
    1. Calculate autocorrelation for group `i` concurrently. Zero delay register is multiplied for each value in this bin group. This can happen concurrently because the zero delay register can be broadcasted to all of the CUDA cores within the same warp.
    2. Add calculated autocorrelation to the corresponding cell in the output array.
    3. Shift the `i`-th group: `shift(sensor, i)`.
* **Completion:** Happens for sure concurrently for data from each sensor on the same CUDA block.
  1. Copy back data from shared memory to global memory



## Tools

## Conclusion

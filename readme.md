# AutocorrelationCUDA
**Student:** Lapo Falcone
**Supervisor:** Gianpaolo Cugola

## Objectives
_The goal of the project was to write a software algorithm to calculate the autocorrelation function extremely efficiently._

### History
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
* The frequency of each photosensor is 80MHz, so the period is 12.5 nanoseconds).
* Each photosensor of the matrix send us how many photons it detected in 128 periods (1600 nanoseconds). So each value fits in 1 byte, since it is a number between 0 and 128.
* For scientific purposes it is needed to calculate the autocorrelation up to 10000 lags.


#### First approach
The first approach to write the algorithm was the trivial one: we tried to calculate `y(n) * y(n-lag)` for each instant in time for each sensor. The pro of this approach is that the code is really simple and elegant, the con is that GPUs are fast, but not this fast. Indeed, despite the fact that we asked the GPU to perform a series of simple additions and multiplications, the throughput of the GPU wasn't high enough, and we ended up processing data 3000 times slower than our target. In numbers, we set-up an environment to calculate the autocorrelation up to 10000 lags of a signal made up of 625000 values. We expected the program to take 1 second to execute. It took 3 seconds, and it analyzed only data from 1 sensor. 3 times 1024 times slower that our goal.

## Tools

## Conclusion

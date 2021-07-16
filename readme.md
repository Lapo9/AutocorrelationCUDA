# AutocorrelationCUDA
**Student:** Lapo Falcone

**Supervisor:** Gianpaolo Cugola

## Background
[Fluorescence correlation spectroscopy (FSC)][1] is a technique used by scientists to study the behavior of biomolecules. The main advantage of this kind of analysis is to be able to monitor the experiment in real-time. The main disadvantage is the same. Indeed, in order to collect and process data as the experiment goes on, using highly specialized hardware components is mandatory. This significantly raises the costs to conduct the experiments, denying to many non-commercial scientific hubs the possibility to contribute to scientific progress, and to form new professionals.

The progress in commercial hardware that took place during the last decade, led many software developers and scientists to start thinking about a different solution: implement the well known compute-intense algorithms, which used to run on dedicated hardware, in a software application running on a standard PC. This not only allowed a faster distribution of the new iterations of the tools involved in scientific analysis, but also did cut down the costs to perform the experiments. First and foremost, a standard, off-the-shelf PC is way cheaper than a custom ASIC or FPGA. Second, a PC is less specialized than an integrated circuit, and it can be used for a wide range of studies.

In particular, the availability of high performance, low cost GPUs and their use not only for graphics processing but for general processing (GPGPU) played the most important role in the regard of this transition from highly specialized hardware to software. Indeed GPUs are highly parallel pieces of hardware that while originally born to satisfy the need of graphic processing may nowadays be used to solve more general problems. In particular, through GPUs it is possible to analyze huge amount of data in parallel, ensuring high throughput, which is key in several scientific experiments, including FSC.

On the other hand GPU programming is not as straight forward as CPU programming, and, until recently, it wasn't open to everyone. The main pitfall of GPU programming lies in the same concept that makes GPUs suitable for scientific purposes: parallelism. The great number of cores a GPU exposes, makes most of the common programming paradigms amiss, and forces the developer to think differently. Memory layout and management becomes a key aspect, and the flow of code must be optimized to avoid divergent execution (namely, avoid if(s)). These aspects will be analyzed in the remainder of this document.

## The experiment
The set-up for a FSC experiment includes three components:
* A matrix of photosensors embedded in a device capable of stimulating raw material with a laser and detecting photons emitted by the material. In our case we assumed a square matrix of 2^n photosensors.
* A server, connected to the device above, where to analyze data provided by the matrix of photosensors.
* A client, where to show the processed data.

Our interest resides on the server, so on the data processing.

The mathematical tool used to process data is the [autocorrelation][5] function, in particular in its discrete form. Simply put, the autocorrelation is a function that correlates a given input to itself. In a more formal way, given a signal `y(t)`, autocorrelation can be defined as: `A(lag) = summation[n in N](y(n) * y(n ± lag))`. It is important to note that in the last factor the presence of `n-lag` is conceptually equivalent to `n+lag`.

So a big value at a specific lag (let's call it `L`) means that the signal `y(t)` tends to be periodic, and has `L` as period. For example, the autocorrelation of `sin(x)` would yield a maximum at `2pi` and its multiples. The autocorrelation tells us how much "periodic" a function is at a given "period". The periodicity of the data collected by the matrix provides valuable information to the scientists who run the experiment.

[_Visualization tool_][6]

### Target
Thanks to the information given us by Prof. Ivan Rech, the responsible of the FSC experiment at Politecnico di Milano, we made this assumptions:
* The matrix is made up of 32x32 photosensors, for a total of 1024 photosensors to process in parallel.
* The frequency of the laser stimulating the raw material (and consequently the frequency photons are revealed by each photosensor) is 80MHz, so the period is 12.5 nanoseconds.
* Each photosensor of the matrix send us how many photons are detected in 128 periods (1600 nanoseconds). So each value fits in 1 byte, since it holds a number between 0 and 128.
* For scientific purposes it is needed to calculate the autocorrelation up to lag 10000.


### First approach
The first approach to write the algorithm was the trivial one: we tried to calculate `y(n) * y(n-lag)` for each instant in time for each sensor. The pro of this approach is that the code is really simple and elegant, the con is that GPUs are fast, but not this fast. Indeed, despite the fact that we asked the GPU to perform a series of simple additions and multiplications, the throughput of the GPU wasn't high enough, and we ended up processing data 3000 times slower than our target. In numbers, we set-up an environment to calculate the autocorrelation up to lag 10000 of a signal made up of 625000 values. We expected the program to take 1 second to execute for 1024 sensors (`80MHz / 128`). It took 3 seconds, and it analyzed only data from 1 sensor. `3 * 1024` times slower that our goal.

At this point it was clear that the approach was wrong, so we decided to look for a different way to calculate the autocorrelation of a discrete signal.

### Multi-tau approach
[After some research][3], we found out that it was possible to calculate an approximative form of the autocorrelation using a multi-tau approach. The fundamental drawback of this technique is that it calculates the by-definition autocorrelation only for lag values smaller than an arbitrary threshold. Then it calculates an approximate form of the autocorrelation, and the higher the lag value is, the less precise the corresponding autocorrelation will be.

The choice of the threshold is critical, since a small threshold guarantees an high throughput, but an higher threshold ensures a more accurate output. Ideally, choosing a threshold tending to infinity leads to an algorithm equivalent to the naive one.

Going a bit more into the details, as we mentioned in the matrix specifications, data is sent to the server as 8 bits integers, each containing the number of photons detected in a 1600ns period. The algorithm calculates the output as `y(n) * y(n-lag)` as long as `lag` is less than the chosen threshold (`Th`). For lag values higher than `Th` and smaller than `Th + 2Th`, the algorithm coalesce 2 periods into a single one, namely creating a single 3200ns period, which contains the sum of the values contained in the original periods. While autocorrelating lag values in the interval `Th < lag < 3Th`, the program uses these longer periods (also called bins), consequently being able to calculate the autocorrelation for lag values in a `3Th - Th = 2Th` interval in the same time it took to compute the by-definition autocorrelation for the first `Th` lag values, doubling the throughput (the drawback resides in the resolution of the output). Then the process repeats for lag values in the interval `3Th < lag < 3Th + 4Th`, by coalescing, each time, 2 3200ns bins in a single 6400ns bin, and so on.

**Lag interval** | **Bin resolution** | **Coalesced bins**
---|---|---
0 ~ Th | 1600ns | 1
Th ~ 3Th | 3200ns | 2
3Th ~ 7Th | 6400ns | 4
7Th ~ 15Th | 12800ns | 8
... | ... | ...

#### Multi-tau algorithm
In order to implement the formal algorithm in an efficient way, we had to create an ad-hoc data structure.

##### Bin group multi sensor memory
This data structure has the responsibility to hold data while being processed, and to maintain this data following the multi-tau rules. So it is its responsibility to coalesce the bins based on the threshold.

![Memory layout](https://github.com/Lapo9/AutocorrelationCUDA/blob/multi_tau/documentation/Images/BinGroup.png)

There is such structure for each sensor of the matrix.

Particular attention should be paid to the zero delay registers. Since data in a bin group different from the 1st is coalesced, as explained in the previous paragraph, also the value corresponding to the current point in time should be coalesced. In a more formal way, for example, in the 2nd bin group we have values made up of the sum of 2 original values: `y(n-lag) + y(n-lag-1)`. In order to have the same "distance" in time between the current point in time and these coalesced values, we must also coalesce the current instant in time. So the autocorrelation for the second group becomes: `(y(n) + y(n-1)) * (y(n-lag) * y(n-lag-1))`, where the first addendum is hold by the zero delay register of the 2nd bin group. It is clear that the first zero delay register simply holds the last value added to the structure.

Moreover the first cell of each group is called accumulator, and it is the place where 2 values from the previous bin group are summed together to obtain the correct bin size for a specific group.

The [interface][7] of the structure is the following:
* `insertNew(sensor, value)`: Inserts the new value in the first position of the first bin group and in the first zero delay register. Moreover it adds the new value to the zero delay registers of all the remaining bin groups.
* `shift(sensor, binGroup)`: Shifts all of the values in the specified bin group one position to the right. The value previously present in the rightmost position leaves the group, and is added to the accumulator of the next group. Accumulator and zero delay register of the group are cleared.
* `get(sensor, group, position)`
* `getZeroDelay(sensor, group)`

In order to improve performance, 2 optimization routes were taken:
* **Reverse shift:** The shift method actually doesn't move all of the values to the right, but actually moves the logical first position of the array to the left. In order to keep track of all of the first positions, a new array is allocated. This optimization forced another improvement: since from now on we had to work with circular arrays, we must use modulo operator, which is very expensive. Luckily it can be optimized if the right-hand-side operand is a power of 2: `x % y = x & (y-1)`.
* **Bank conflicts avoidance:** In order to use the potential of CUDA architecture fully, we arranged the arrays relative to the bin groups of each sensor "vertically", instead of "horizontally". So, after the last position of the first bin group of the first sensor, there is the first position of the first bin group of the second sensor, and not the first position of the second group of the first sensor. This way it was impossible for cells accessed concurrently to end up on the same bank.

![Memory layout](https://github.com/Lapo9/AutocorrelationCUDA/blob/multi_tau/documentation/Images/BinGroupMultiSensorMemory.png)

###### Actually calculated lag values formula
Since the multi-tau approach unties the number of operations from the actual number of calculated lag values, it is important to have a formula to get how many lag values are calculated given the number of bin groups `B`, and the size of a bin group `S`: `number of lag values = summation[n: 0 -> B-1](S * 2^n)`.

For example, given 10 bin groups of size 32, we have: `number of lag values = summation[n: 0 -> 9](32 * 2^n) = 32736`.

This formula can be obtained by thinking about the meaning of "bin group". The first bin group has single bins, so, looking at the example, it calculates the first `32 * 1 = 32 * 2^0 = 32` lag values. The second bin group has double bins (bins with half the resolution, look at the table above), so it calculates `32 * 2 = 32 * 2^1 = 64` lag values. So with two groups we calculate `32 + 64 = 96` lag values. The third group has bins with a resolution 4 times smaller than the base one, so it computes `32 * 4 = 32 * 2^2 = 128` lag values. With 3 bin groups we calculate `32 + 64 + 128 = 224` lag values. And so on.


##### Algorithm
The algorithm was thought to maximize the number of operations to run concurrently. Since, in order to compute autocorrelation for bin group `x`, bin group `x-1` must be finished, we ended up to parallelize on the computation of autocorrelation of values on the same bin group. We chose the critical size of the bin group being 32, which turned out to be the right compromise between high throughput and output accuracy. Since we were bound to 1024 sensors, it means that 32768 multiplications and additions could run in parallel. This is way more than the number of CUDA cores in any of the most recent Nvidia GPUs, so this level of concurrency alone maximized the occupancy of any modern GPU.

We then chose to create 2D CUDA blocks sized 32*8. This means that on the x coordinate we have CUDA cores which work concurrently on the calculation of different lag values for the same sensor. In the y coordinate we could handle 8 sensors per block. The size of the CUDA block is then 256, which turned out to maximize the number of registers and shared memory available on each block.

Last, the choice of 32 as the size of a bin group also makes it possible to have all of the threads working on the same warp. This gains particular importance during the computation of the autocorrelation (see Computation.3.1).

* **Set-up:** Each CUDA block allocates and copy only data from sensors residing on that CUDA block.
1. Allocate array for output and bin group multi sensor memory on shared memory.
2. Copy output and bin group multi sensor memory passed as inputs (from previous call to this function), from global memory to shared memory.
* **Computation:** Loop that repeats for each value in the input array. Next steps happen for sure concurrently for data from each sensor residing on the same CUDA block.
1. Insert new datum from input array to bin group multi sensor memory: `insertNew(sensor, value)`.
2. Compute how many bin groups have to be calculated during this iteration (`T`). The first bin group is calculated on every iteration, the second one once yes and once no, the third one once every 4 iterations, and so on. This happens because a successive bin group has half the resolution of the previous one.
3. `for i < T`

   1. Calculate autocorrelation for group `i` concurrently. Zero delay register is multiplied for each value in this bin group. This can happen concurrently because the zero delay register can be broadcasted to all of the CUDA cores within the same warp.
   2. Add calculated autocorrelation to the corresponding cell in the output array.
   3. Shift the `i`-th group: `shift(sensor, i)`.

* **Completion:** Happens for sure concurrently for data from each sensor on the same CUDA block.
1. Copy back data from shared memory to global memory


## Tools
_Nvidia CUDA library, C++_

### Nvidia CUDA library
In order to program on the GPU we decided to use the library provided by Nvidia: [CUDA][2]. CUDA is exposed through a C API, and only works on Nvidia devices. We chose CUDA because the physical machine used for the experiment at Politecnico mounts a Titan X (Pascal).

GPU programming, as mentioned before, has a different paradigm compared to standard CPU programming. These are the most important concepts that differentiate the two.

#### Programming model
Kernels are functions that run on GPU. Unlike standard functions, where multithreading must be specifically implemented, kernels are multithreaded by default. This means that by default each instruction present in a kernel is executed simultaneously by all of the threads the kernel is made up of. Each thread can be easily identified by its unique ID, and, based on this ID, a thread can execute the same instruction as the other threads, on different data. This concept is known as SIMD, Single Instruction Multiple Data.

On CUDA architecture each thread is executed by a CUDA core. Moreover threads can be organized in a thread hierarchy. Indeed, at kernel invocation, it is possible to specify how many threads and how many threads blocks are available. A thread block is a set of threads, which have incremental IDs. Threads in threads blocks can also be organized in a 2D or 3D way. If one decides to use a 3D block, each thread is identified by a x, y and z component. By the way, the shape of the block is irrelevant performance wise, since even in a 2D block threads are organized in a linear fashion, and their real ID can be obtained by: `ID = ID.x + ID.y * blockDim.x`.

Even threads block can be organized in a structure called grid. A grid is a set of thread blocks, just like a thread block is a set of threads. The structure is the same, and even grids can be organized in a 2D or 3D fashion.

At a less abstract level, the SIMD paradigm happens between groups of 32 threads, regardless of the block or grid size. A set of 32 contiguous threads is a warp. When a kernel is launched its blocks are assigned to an entity called Streaming Multiprocessor (SM). A SM has one or more schedulers which map logical warps to group of 32 CUDA cores (also called streaming processors (SP)). Each thread in a warp has a specific program counter, so it can execute whatever instruction it wants. On the other hand each thread on a warp must execute the same instruction at a time or being idle, so it is important that threads within a single warp all execute the same instruction. If this doesn't happen, threads are divergent. Divergent threads within a warp cannot execute concurrently, so divergence avoidance is a key aspect in GPU programming.

Obviously more warps execute concurrently on the same or on different SM.

#### Memory model
Another fundamental aspect of CUDA is the memory hierarchy. Since different kind of memories has read/write throughput that differs by orders of magnitude, organize data in an optimized way is key.

##### Global memory
Global memory is the slowest but most capacious memory. When allocating or copying data from CPU to GPU, it ends up in global memory. Each time a kernel tries to read or write something to global memory, a set of 128 contiguous bytes is read/written, regardless of the actual size of the read/write. It is thus important to try to coalesce memory accesses and to keep memory aligned to 128 bytes. Indeed, if data to be read/written is straddling two groups of 128 bytes, more than one transaction is needed, even though the requested data is less than 128 bytes long. Global memory is cached, and it is possible to specify certain segments to be cached more or less aggressively. It is persistent across different kernel invocations.

##### Constant memory
Constant memory resides on the same chip as global memory, but it is cached more aggressively on an ad-hoc constant cache. As the name suggest constant memory is read only.

##### L2 cache
L2 cache is the slowest cache available, but it is way faster than global memory. It isn't persistent across different kernel invocations.

##### L1 cache
L1 cache is an on chip cache, and it is the fastest cache available. L1 cache is placed on SM, so each block shares the L1 cache.  It isn't persistent across different kernel invocations.

##### Shared memory
Shared memory resides on the L1 cache. Indeed L1 cache can be customized to host 1/3 of L1 automatic cache and 2/3 of shared memory and vice-versa. Shared memory is considered a manual L1 cache. A key aspect of managing shared memory is to avoid bank conflicts. Shared memory is organized in 32 banks. A bank is a memory area that holds 4 byte words. If different threads in the same warp try to access different words on the same bank, access is sequential. If they try to access different bytes belonging to the same word, or words placed on different banks, access is concurrent.

![Shared memory layout](https://github.com/Lapo9/AutocorrelationCUDA/blob/multi_tau/documentation/Images/SharedMemory.png)

##### Registries
Each CUDA core has its own registries, where local variables are placed. If more local variables are instantiated than the number of registries, registry spill occurs. Data that cannot be placed on registries is thus saved on local memory.

##### Local memory
Placed on the same chip as global memory.

##### Texture memory
Global memory optimized for 2D access.


**Memory** | **[Bandwidth][4]** | **Persistent** | **On chip** | **Peculiarities**
---|---|---|---|---
Global | 510 GB/s | Yes | No | Coalesced access
Constant | 510 GB/s | Yes | No | Read only
L2 cache | 1624 GB/s | No | No |
L1 cache | 7700 GB/s | No | Yes | Shared with shared memory
Shared | 7700 GB/s | No | Yes | Bank conflicts
Registries | - | No | Yes | Can spill
Local | 510 GB/s | No | No | Expand registries
Texture | 510 GB/s | Yes | No | Optimized for 2D access

### CUDA profiler
In order to study the behavior of the algorithm and its performance, we used the Nvidia visual profiler (nvvp). The tool shows in a timeline all of the API function and kernel calls, and it allows for a very in-depth analysis of fundamental aspects of the execution. The metrics we mostly took into consideration are the following:
* **Occupancy:** How many of the available CUDA cores are utilized on average. An high occupancy means more operations are performed concurrently, but having a 100% occupancy not always is the best way to improve performance. Indeed, in memory intensive algorithm, it is often preferred to use a big chunk of shared memory. Since shared memory resides on SM, it means that the more shared memory a block uses, the less concurrent warps can be executed, thus reducing occupancy. At the end of the day our algorithm achieves about 60% occupancy.
* **Global efficiency:** As mentioned before, global memory access are coalesced. Global efficiency measures the ratio of requested bytes on read/written bytes. We got 99% global read efficiency, but only 29% global store efficiency. On the other hand almost 230000 read transactions were requested, and only 21000 store, so the low store global efficiency doesn't affect performance severely, but it is for sure improvable.
* **Warp efficiency:** Average ratio between the active threads in a warp over the maximum active threads in a warp (32). It basically measures divergence. We got 77% efficiency, which cannot be improved due to the fact that some operations must be executed only a smaller number of times. Indeed on one block we have 256 threads, which all run concurrently during the actual autocorrelation computation phase, but only 8 threads are responsible to insert the new value into the bin group multi sensor memory.

Since the Titan X was installed on a remote machine, in order to use the profiler on it we had to use SSH to access the remote host with X11 forwarding to get the nvvp GUI. To achieve this from the Windows 10 machine I was developing on, I had to use [PuTTY][8] and [Xming][9].

### Excel
In order to verify that the results yield by the algorithm were correct, we created a file in Microsoft Excel that computes the autocorrelation following the multi-tau approach. In the Excel file only a simplified version of the algorithm is present, which is only able to calculate the autocorrelation for 3 bin groups of 32 values for one single sensor. This is because we only need to verify the general correctness of the algorithm, and not the edge cases, that were checked manually. The file can be found [on the GitHub repository][10].

## Conclusion
At the end of the development the goal was reached. Our final test is structured like this:
* The input is randomly generated and saved in a vector (notice that the actual values that compose our input do not impact the performance of the algorithm, so it is acceptable to use random values to measure performance).
* The input is made up of 100 values for each sensor for each execution of the kernel.
* The kernel is called 1000 times.
* There are 10 bin groups for each sensor. This means that we calculated 32736 lag values, three times the minimum requirement of 10000. For calculation look at "Actually calculated lag values formula".
* A low overhead timer object measures the duration of each kernel execution.
* Code was compiled with [this makefile][11].

The results were even better than the target. The average execution time of the kernel was 150 microseconds. This means that in 1 second we are able to compute the autocorrelation for more than 666000 instants for each sensor (`100 * (1 / 150us)`), which is more than the 625000 that was our original goal, as discussed before.

### Improvements
We think it is still possible both to go faster and improve the code.

To increase the performance it is possible to study more deeply the access pattern both to the global and to the local memory. Especially the store instructions to global memory can probably be coalesced better. Another improvement would be to put the input data into the constant memory, even though it wouldn't reduce the execution time as dramatically as the previous optimization.

Concerning the C++ code, it is possible to work on its flexibility. Right now the parameters of the algorithm, such as the number of sensors or the number of bin groups, are hard coded into the Definitions.h file. This is because it is fundamental that these values are known at compilation time, indeed they are all coded as `constexpr`. It is probably possible to use template metaprogramming to increase flexibility and generality of code.

### Final personal thoughts
By a personal point of view, I can say I'm really satisfied with this project.

At the beginning I was looking to a project to improve my C++ knowledge, and I wasn't so sure this project would have helped me. But as time went I started appreciating what I was doing more and more, both because I was indeed improving my C++ experience, and because I was learning a completely new programming paradigm, which will be for sure more common in the future, given the tendency to use more and more parallel programming techniques. I also enjoyed learning about CUDA architecture, and programming towards it, something I've never done before, since I've always programmed non performance-critical code. It taught me a lot about memory and branch management, and I was very surprised to know that you can write elegant code even in performance critical sections, and not hacks.

Last I learned how to work with another person and not alone, and this is probably the most valuable teaching I got. For this reason I'd like to say thank you to Prof. Gianpaolo Cugola, who has always been really helpful and treated me as a coworker more than as his student.


[1]: https://autocorrelationcuda.neocities.org/non_doxygen/PhD_Gong_Sixia.pdf
[2]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
[3]: https://aip.scitation.org/doi/full/10.1063/1.2721116
[4]: https://arxiv.org/pdf/1804.06826.pdf
[5]: https://en.wikipedia.org/wiki/Autocorrelation
[6]: https://www.desmos.com/calculator/q4hnqeab6n?lang=it
[7]: https://autocorrelationcuda.neocities.org/index.html
[8]: https://www.putty.org/
[9]: http://www.straightrunning.com/XmingNotes/
[10]: https://github.com/Lapo9/AutocorrelationCUDA/blob/1c79edcd8c3ba3c14ac7fc10030f4a51cba21f7f/documentation/Autocorrelation_calculator.xlsx
[11]: https://github.com/Lapo9/AutocorrelationCUDA/blob/1c79edcd8c3ba3c14ac7fc10030f4a51cba21f7f/Makefile

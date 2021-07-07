#ifndef AUTOCORRELATIONCUDA_LINUXTIMER
#define AUTOCORRELATIONCUDA_LINUXTIMER

#include <vector>
#include <functional>
#include <chrono>

namespace AutocorrelationCUDA{

/**
* @brief Simple timer class to measure performance.
**/
class Timer final {

	public:

	/**
	* @brief Creates a new Timer, with the specified final action and strategy to get the current time.
	* @param finalAction Action to be performed when the timer is destroyed (for example save collected data to file). The intervals calculated are passed in as a vector of double.
	* @param getCurrentTime Function to get the current time, used to calculate intervals.
	**/
	Timer(std::function<void(std::vector<double>)> finalAction, std::function<double()> getCurrentTime) : finalAction{finalAction}, getCurrentTime{getCurrentTime} {}
	
	/**
	* @brief Performs the final action and destroys the Timer.
	**/
	~Timer() {
		results.emplace_back(getCurrentTime() - startTime);
		finalAction(results);
	}

	Timer(Timer&) = delete;
	Timer(Timer&&) = delete;
	Timer operator= (Timer&) = delete;
	Timer operator= (Timer&&) = delete;


	/**
	* @brief Starts the Timer.
	**/
	void start() {
		startTime = intervalTime = getCurrentTime();
	}


	/**
	* @brief Gets the split time and saves the elapsed time from last call to this function (or start) to the result array.
	**/
	void getInterval() {
		double stopTime = getCurrentTime();
		results.emplace_back(stopTime-intervalTime);
		intervalTime = stopTime;
	}


	/**
	* @brief Starts a new interval without saving previous one.
	**/
	void startInterval() {
		intervalTime = getCurrentTime();;
	}



	private:

	double startTime;
	double intervalTime;
	std::vector<double> results;
	std::function<double()> getCurrentTime;
	std::function<void(std::vector<double>)> finalAction;
};

}

#endif


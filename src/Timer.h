#ifndef AUTOCORRELATIONCUDA_LINUXTIMER
#define AUTOCORRELATIONCUDA_LINUXTIMER

#include <vector>
#include <functional>
#include <chrono>

namespace AutocorrelationCUDA{

class Timer final {

	public:

	Timer(std::function<void(std::vector<double>)> finalAction, std::function<double()> getCurrentTime) : finalAction{finalAction}, getCurrentTime{getCurrentTime} {}
	
	~Timer() {
		results.emplace_back(getCurrentTime() - startTime);
		finalAction(results);
	}

	Timer(Timer&) = delete;
	Timer(Timer&&) = delete;
	Timer operator= (Timer&) = delete;
	Timer operator= (Timer&&) = delete;


	void start() {
		startTime = intervalTime = getCurrentTime();
	}


	void getInterval() {
		double stopTime = getCurrentTime();
		results.emplace_back(stopTime-intervalTime);
		intervalTime = stopTime;
	}


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


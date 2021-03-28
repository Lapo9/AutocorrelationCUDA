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
		finalAction(results);
	}

	Timer(Timer&) = delete;
	Timer(Timer&&) = delete;
	Timer operator= (Timer&) = delete;
	Timer operator= (Timer&&) = delete;


	void start() {
		startTime = getCurrentTime();
	}


	void getInterval() {
		double interval = getCurrentTime();
		results.emplace_back(interval);
		startTime = interval;
	}



	private:

	double startTime;
	std::vector<double> results;
	std::function<double()> getCurrentTime;
	std::function<void(std::vector<double>)> finalAction;
};

}

#endif


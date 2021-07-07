#ifndef AUTOCORRELATIONCUDA_FEEDER
#define AUTOCORRELATIONCUDA_FEEDER

#include <istream>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iostream>

namespace AutocorrelationCUDA {

using namespace std::chrono_literals;

/**
* @brief Once every x milliseconds, the Feeder runs the user defined transfer() function.
**/
class Feeder final {

	public:
	
	/**
	* @brief Creates a Feeder with the specified rest time and function.
	* @param rest How many milliseconds must elaps between two calls to the transfer function.
	* @param transfer Routine to execute once every rest milliseconds.
	**/
	Feeder(const std::chrono::milliseconds rest, const std::function<void()> transfer) : transfer{transfer}, rest{rest} {}

	Feeder(const Feeder& feeder) = delete;
	Feeder(Feeder&& feeder) = delete;
	
	Feeder& operator = (const Feeder& feeder) = delete;
	Feeder& operator = (Feeder&& feeder) = delete;

	/**
	* @brief Stops the Feeder and destroys it.
	**/
	~Feeder() noexcept {
		setTerminate(true);
		cv.notify_all();
		transferData.join();
	}


	/**
	* @brief Starts the thread that loops and run the user defined transfer() function.
	**/
	void start() {
		std::cout << "\nSTART\n";
		//a Feeder can be start only once
		if(firstStarted) {
			throw std::runtime_error("Cannot call this function again!");
		}
		firstStarted = true;

		//launch new thread
		transferData = std::thread([this]{
			while (!isTerminate()) {
				transfer(); //call user defined function
				if(rest>0s){
					std::unique_lock<std::mutex> guard{mtx}; //acquire mutex in order to use the condition variable
					cv.wait_for(guard, rest, [this]{return terminated;}); //wait the rest time
					cv.wait(guard, [this]{return !paused || terminated;}); //wait if the thread is paused
				}
			}
			});
	}


	/**
	* @brief Pauses the Feeder till resume() is called.
	**/
	void pause() {
		if(!paused) {
			std::cout << "\nPAUSED\n";
			std::lock_guard<std::mutex> guard{mtx};
			this->paused = true;
			cv.notify_all();
		}
	}

	/**
	* @brief Resumes a paused Feeder.
	**/
	void resume() {
		if(paused) {
			std::cout << "\nRESUME\n";
			std::lock_guard<std::mutex> guard{mtx};
			this->paused = false;
			cv.notify_all();
		}
	}


	private:

	//access terminate member synchronously
	bool isTerminate() {
		std::lock_guard<std::mutex> guard{mtx};
		return terminated;
	}

	//set terminate member synchronously
	void setTerminate(bool terminated) {
		std::lock_guard<std::mutex> guard{mtx};
		this->terminated = terminated;
		cv.notify_all();
	}



	std::function<void()> transfer;
	bool terminated = false;
	bool paused = false;
	bool firstStarted = false;
	std::mutex mtx;
	std::condition_variable cv;
	std::thread transferData;
	std::chrono::milliseconds rest; //time delay between 2 subsequent calls to transfer (0 means no wait)

};
};


#endif

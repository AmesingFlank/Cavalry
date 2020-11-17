#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include "GpuCommons.h"

using TimeType = std::chrono::time_point<std::chrono::system_clock>;

struct TimerEvent{
    TimeType start;
    TimeType end;
    bool finished = false;
};

class Timer
{
    
    public:
        static Timer& getInstance()
        {
            static Timer    instance; 
            return instance;
        }
    private:
        Timer() {}                   

    public:
        Timer(const Timer &)               = delete;
        void operator=(const Timer &)  = delete;


public:
    TimeType now(){
        return std::chrono::system_clock::now();
    }

    void start(const std::string& name)
    {
        if(events.find(name)!=events.end()){
            SIGNAL_ERROR("event already exists");
        }
        TimerEvent event;
        event.start = now();
        events[name] = event;
    }
    
    void stop(const std::string& name)
    {
        if(events.find(name)==events.end()){
            SIGNAL_ERROR("event doesn't exists");
        }
        TimerEvent& event = events.at(name);
        if (event.finished) {
            SIGNAL_ERROR("event already finished");
        }
        event.finished = true;
        event.end = now();
    }

    void printStatistics() const {
        for(auto event:events){
            double milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(event.second.end - event.second.start).count();
            std::cout<<event.first<<"  took   "<<milliseconds<<"ms"<<std::endl;
        }
    }

    void printStatistics(const std::string& name) const {
        if (events.find(name) == events.end()) {
            SIGNAL_ERROR("Timer event not found : %s\n",name.c_str());
        }
        const TimerEvent& event = events.at(name);
        if (!event.finished) {
            SIGNAL_ERROR("event not yet finished. cannot print stats.");
        }
        double milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(event.end - event.start).count();
        std::cout << name << "  took   " << milliseconds << "ms" << std::endl;
    }
    

private:
    std::unordered_map<std::string,TimerEvent> events;
};
cmake_minimum_required(VERSION 3.29)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_library(NeuralNetwork SHARED
        Network.cpp
        Network.h
        Serialize.cpp
        QLearning.cpp)

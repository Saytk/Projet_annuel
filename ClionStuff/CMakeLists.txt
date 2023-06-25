    cmake_minimum_required(VERSION 3.23)
    project(projetAnnuel)

    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/C:/Users/erwan/OneDrive/Bureau/UnityProjetcs/PAClient/Assets/Plugins)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/C:/Users/erwan/OneDrive/Bureau/UnityProjetcs/PAClient/Assets/Plugins)

    add_executable(projetAnnuel main.cpp Hoge.cpp Hoge.h)
    add_library(PAMLDLL SHARED Hoge.cpp Hoge.h)

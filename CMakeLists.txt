cmake_minimum_required(VERSION 3.26)
project(correccion_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)


enable_language(CXX CUDA)

find_package(fmt)
find_package(SFML)

add_executable(correccion_cuda
        main_cuda.cpp
        kernel.cu
        )

target_link_libraries(correccion_cuda fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(correccion_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(correccion_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(espejo espejo.cpp espejo.cu)

target_link_libraries(espejo fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(espejo PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(espejo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(suma  suma_cuda.cpp  suma.cu)

target_link_libraries(suma fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(suma PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(suma PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(rotacion rotacion_cuda.cpp rotacion.cu)

target_link_libraries(rotacion fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(rotacion PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(rotacion PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(sobel sobel_main.cpp sobel.cu)

target_link_libraries(sobel fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(sobel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(sobel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)



add_executable(mediana  mediana_cuda.cpp mediana.cu)

target_link_libraries(mediana fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(mediana PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(mediana PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)




